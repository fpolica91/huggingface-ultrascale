"""
Step 4: DP with All Three Optimizations
=========================================
What changed from 03_bucket_dp.py:
- Added gradient accumulation (multiple forward/backward before optimizer step)
- Only sync gradients on the FINAL accumulation step (no_sync on others)
- This is now a complete data parallel implementation with all 3 optimizations:
    Opt 1: Overlap all-reduce with backward (hooks)
    Opt 2: Bucket gradients for efficient communication
    Opt 3: Skip sync on intermediate accumulation steps (no_sync)

Run with:
  torchrun --nproc_per_node=8 04_full_dp.py
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
    print("Step 4: DP with all 3 optimizations")
    print("=" * 60)


# ============================================================
# Same Bucket classes from step 3
# ============================================================
class GradBucket:
    def __init__(self, params, world_size):
        self.params = list(params)
        self.world_size = world_size
        self.total_size = sum(p.numel() for p in self.params)
        self.flat_grad = torch.zeros(self.total_size, device=self.params[0].device)
        self.param_slices = {}
        offset = 0
        for p in self.params:
            size = p.numel()
            self.param_slices[p] = (offset, offset + size)
            offset += size
        self.ready_count = 0
        self.handle = None
    
    def mark_ready(self, param):
        start, end = self.param_slices[param]
        self.flat_grad[start:end].copy_(param.grad.data.flatten())
        self.ready_count += 1
        if self.ready_count == len(self.params):
            self.flat_grad /= self.world_size
            self.handle = dist.all_reduce(
                self.flat_grad, op=dist.ReduceOp.SUM, async_op=True
            )
    
    def wait_and_copy_back(self):
        if self.handle is not None:
            self.handle.wait()
        for p in self.params:
            start, end = self.param_slices[p]
            p.grad.data.copy_(self.flat_grad[start:end].view_as(p.grad))
    
    def reset(self):
        self.ready_count = 0
        self.handle = None
        self.flat_grad.zero_()


class BucketManager:
    def __init__(self, model, world_size, bucket_size_mb=25):
        self.world_size = world_size
        bucket_size = bucket_size_mb * 1024 * 1024 // 4
        all_params = [p for p in model.parameters() if p.requires_grad]
        all_params = list(reversed(all_params))
        
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
        
        self.param_to_bucket = {}
        for bucket in self.buckets:
            for p in bucket.params:
                self.param_to_bucket[id(p)] = bucket
    
    def mark_param_ready(self, param):
        bucket = self.param_to_bucket[id(param)]
        bucket.mark_ready(param)
    
    def wait_all(self):
        for bucket in self.buckets:
            bucket.wait_and_copy_back()
    
    def reset(self):
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
bucket_mgr = BucketManager(model, world_size, bucket_size_mb=1)

# ============================================================
# THE KEY ADDITION: no_sync flag
# ============================================================
# This flag controls whether hooks fire all-reduce or not.
# During intermediate accumulation steps: False (no communication)
# On the final accumulation step: True (fire all-reduce)

require_backward_grad_sync = True  # will be toggled per accumulation step

def hook_fn(param):
    """Only sync when require_backward_grad_sync is True."""
    if require_backward_grad_sync:
        bucket_mgr.mark_param_ready(param)

for param in model.parameters():
    if param.requires_grad:
        param.register_post_accumulate_grad_hook(hook_fn)

# ============================================================
# Training config
# ============================================================
micro_batch_size = 32
grad_acc_steps = 4  # accumulate 4 micro-batches before syncing

if rank == 0:
    print(f"Micro-batch size per GPU: {micro_batch_size}")
    print(f"Gradient accumulation steps: {grad_acc_steps}")
    print(f"Effective batch per GPU: {micro_batch_size * grad_acc_steps}")
    print(f"Global batch size: {micro_batch_size * grad_acc_steps * world_size}")
    print(f"  = mbs({micro_batch_size}) × grad_acc({grad_acc_steps}) × dp({world_size})")
    print(f"Buckets: {len(bucket_mgr.buckets)}")
    print("=" * 60)

# ============================================================
# Training loop — all 3 optimizations working together
# ============================================================
num_steps = 5

for step in range(num_steps):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    
    optimizer.zero_grad()
    accumulated_loss = 0.0
    
    for acc_step in range(grad_acc_steps):
        # Generate different data for each accumulation step
        data = torch.randn(micro_batch_size, 1024, device=device)
        targets = torch.randn(micro_batch_size, 512, device=device)
        
        # ===== OPTIMIZATION 3: no_sync on intermediate steps =====
        # Only sync on the LAST accumulation step
        is_last_acc_step = (acc_step == grad_acc_steps - 1)
        require_backward_grad_sync = is_last_acc_step
        
        if is_last_acc_step:
            bucket_mgr.reset()  # prepare buckets for the sync step
        
        # Forward
        outputs = model(data)
        loss = nn.functional.mse_loss(outputs, targets) / grad_acc_steps
        # ^ divide by grad_acc_steps so accumulated gradients are averaged
        
        # Backward
        # Intermediate steps: hooks see require_backward_grad_sync=False → no all-reduce
        # Final step: hooks see True → buckets fill → all-reduce fires (opt 1+2)
        loss.backward()
        
        accumulated_loss += loss.item()
    
    # Wait for bucket all-reduces from the final accumulation step
    bucket_mgr.wait_all()
    
    # Optimizer step — all GPUs have identical averaged gradients
    optimizer.step()
    
    end.record()
    torch.cuda.synchronize()
    
    step_time = start.elapsed_time(end)
    
    if rank == 0:
        print(f"Step {step}: loss={accumulated_loss:.4f}, time={step_time:.2f}ms")

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
        print("\nAll 3 optimizations active:")
        print("  Opt 1: Hooks overlap all-reduce with backward ✓")
        print("  Opt 2: Gradients bucketed for efficient communication ✓")
        print("  Opt 3: no_sync skips communication on intermediate accumulation steps ✓")

dist.destroy_process_group()