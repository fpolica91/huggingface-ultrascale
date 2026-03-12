"""
Step 3: DP with Bucketed Gradient Sync (Optimization 1 + 2)
=============================================================
What changed from 02_overlap_dp.py:
- Instead of firing all-reduce per PARAMETER (many tiny communications),
  we group parameters into BUCKETS and fire one all-reduce per bucket.
- Fewer, fatter communications = more efficient NVLink utilization.
- Bucket fires its all-reduce when ALL params in that bucket have gradients ready.

Think: packing items into boxes before shipping.
  Many small packages = lots of overhead per package.
  Fewer big boxes = less overhead, better throughput.

Run with:
  torchrun --nproc_per_node=8 03_bucket_dp.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW

dist.init_process_group(backend="nccl")

rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f"cuda:{rank}")
torch.cuda.set_device(device)

if rank == 0:
    print("Step 3: DP with bucketed gradient sync")
    print("=" * 60)


# ============================================================
# Bucket class — groups parameters for efficient all-reduce
# ============================================================
class GradBucket:
    """
    Collects gradients from multiple parameters into one flat tensor.
    When all parameters in the bucket have their gradients ready,
    fires a single async all-reduce on the whole bucket.
    """
    def __init__(self, params, world_size):
        self.params = list(params)
        self.world_size = world_size
        
        # Calculate total size needed
        self.total_size = sum(p.numel() for p in self.params)
        
        # One flat tensor to hold all gradients in this bucket
        self.flat_grad = torch.zeros(self.total_size, device=self.params[0].device)
        
        # Map each param to its slice of the flat tensor
        self.param_slices = {}
        offset = 0
        for p in self.params:
            size = p.numel()
            self.param_slices[p] = (offset, offset + size)
            offset += size
        
        self.ready_count = 0
        self.handle = None  # async all-reduce handle
    
    def mark_ready(self, param):
        """Called when a parameter's gradient is computed during backward."""
        start, end = self.param_slices[param]
        # Copy gradient into the flat buffer
        self.flat_grad[start:end].copy_(param.grad.data.flatten())
        self.ready_count += 1
        
        # When ALL params in bucket are ready, fire all-reduce
        if self.ready_count == len(self.params):
            self.flat_grad /= self.world_size
            # async_op=True → non-blocking, NVLink works while GPU continues
            self.handle = dist.all_reduce(
                self.flat_grad, op=dist.ReduceOp.SUM, async_op=True
            )
    
    def wait_and_copy_back(self):
        """Wait for all-reduce to finish, then copy averaged grads back to params."""
        if self.handle is not None:
            self.handle.wait()
        
        for p in self.params:
            start, end = self.param_slices[p]
            p.grad.data.copy_(self.flat_grad[start:end].view_as(p.grad))
    
    def reset(self):
        """Reset for next iteration."""
        self.ready_count = 0
        self.handle = None
        self.flat_grad.zero_()


# ============================================================
# BucketManager — divides all parameters into buckets
# ============================================================
class BucketManager:
    """
    Splits model parameters into buckets of approximately bucket_size_mb.
    Parameters are added in REVERSE order (matching backward pass order)
    so that the first bucket to fill = the first layers to finish backward.
    """
    def __init__(self, model, world_size, bucket_size_mb=25):
        self.world_size = world_size
        bucket_size = bucket_size_mb * 1024 * 1024 // 4  # float32 = 4 bytes
        
        # Reverse parameter order — backward goes from last layer to first
        all_params = [p for p in model.parameters() if p.requires_grad]
        all_params = list(reversed(all_params))
        
        # Split into buckets
        self.buckets = []
        current_bucket_params = []
        current_size = 0
        
        for p in all_params:
            if current_size + p.numel() > bucket_size and current_bucket_params:
                self.buckets.append(GradBucket(current_bucket_params, world_size))
                current_bucket_params = []
                current_size = 0
            current_bucket_params.append(p)
            current_size += p.numel()
        
        if current_bucket_params:
            self.buckets.append(GradBucket(current_bucket_params, world_size))
        
        # Map each param to its bucket
        self.param_to_bucket = {}
        for bucket in self.buckets:
            for p in bucket.params:
                self.param_to_bucket[id(p)] = bucket
    
    def mark_param_ready(self, param):
        """Called by the hook when a gradient is ready."""
        bucket = self.param_to_bucket[id(param)]
        bucket.mark_ready(param)
    
    def wait_all(self):
        """Wait for all bucket all-reduces to finish and copy gradients back."""
        for bucket in self.buckets:
            bucket.wait_and_copy_back()
    
    def reset(self):
        """Reset all buckets for next iteration."""
        for bucket in self.buckets:
            bucket.reset()


# ============================================================
# Model setup
# ============================================================
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(1024, 2048),
    nn.ReLU(),
    nn.Linear(2048, 2048),
    nn.ReLU(),
    nn.Linear(2048, 512),
).to(device)

optimizer = AdamW(model.parameters(), lr=1e-3)

# Create bucket manager
bucket_mgr = BucketManager(model, world_size, bucket_size_mb=1)

if rank == 0:
    print(f"Created {len(bucket_mgr.buckets)} buckets")
    for i, b in enumerate(bucket_mgr.buckets):
        print(f"  Bucket {i}: {len(b.params)} params, {b.total_size:,} elements")
    print("=" * 60)

# Register hooks — each param notifies its bucket when gradient is ready
for param in model.parameters():
    if param.requires_grad:
        param.register_post_accumulate_grad_hook(
            lambda p: bucket_mgr.mark_param_ready(p)
        )

# Same data
micro_batch_size = 32
data = torch.randn(micro_batch_size, 1024, device=device)
targets = torch.randn(micro_batch_size, 512, device=device)

# ============================================================
# Training loop — buckets handle all-reduce automatically
# ============================================================
num_steps = 5

for step in range(num_steps):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    
    bucket_mgr.reset()
    
    # Forward
    outputs = model(data)
    loss = nn.functional.mse_loss(outputs, targets)
    
    # Backward — hooks fire, buckets accumulate, all-reduce when full
    optimizer.zero_grad()
    loss.backward()
    
    # Wait for all bucket all-reduces to complete
    bucket_mgr.wait_all()
    
    # Optimizer step
    optimizer.step()
    
    end.record()
    torch.cuda.synchronize()
    
    step_time = start.elapsed_time(end)
    
    if rank == 0:
        print(f"Step {step}: loss={loss.item():.4f}, time={step_time:.2f}ms")

# Verify sync
with torch.no_grad():
    first_param = next(model.parameters())
    param_sum = first_param.sum().clone()
    all_sums = [torch.zeros_like(param_sum) for _ in range(world_size)]
    dist.all_gather(all_sums, param_sum)
    if rank == 0:
        all_same = all(torch.allclose(all_sums[0], s) for s in all_sums)
        print("=" * 60)
        print(f"All GPUs have identical weights: {all_same}")

dist.destroy_process_group()