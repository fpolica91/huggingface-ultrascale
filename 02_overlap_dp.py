"""
Step 2: DP with Overlapped Gradient Sync (Optimization 1)
==========================================================
What changed from 01_naive_dp.py:
- Instead of all-reducing AFTER the full backward pass,
  we attach a hook to each parameter that fires all-reduce
  AS SOON as that parameter's gradient is ready.
- This overlaps communication (NVLink) with computation (CUDA cores).

The hook approach: assembly line thinking.
  Backward finishes layer 3 → hook fires all-reduce for layer 3
  Meanwhile backward continues computing layer 2...

Run with:
  torchrun --nproc_per_node=8 02_overlap_dp.py
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
    print("Step 2: DP with overlapped gradient sync")
    print("=" * 60)

# Same model as before
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(1024, 2048),
    nn.ReLU(),
    nn.Linear(2048, 2048),
    nn.ReLU(),
    nn.Linear(2048, 512),
).to(device)

optimizer = AdamW(model.parameters(), lr=1e-3)

# ============================================================
# THE KEY CHANGE: Register hooks for overlapped all-reduce
# ============================================================
# When a gradient is computed during backward pass,
# the hook fires IMMEDIATELY and starts the all-reduce
# while backward continues on earlier layers.

# Attach hook to every parameter that needs gradients
for param in model.parameters():
    if param.requires_grad:
        param.register_post_accumulate_grad_hook(
            lambda p, ws=world_size: (
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM),
                p.grad.div_(ws),
            ) and None
        )

if rank == 0:
    print(f"Registered all-reduce hooks on {sum(1 for p in model.parameters() if p.requires_grad)} parameters")
    print("=" * 60)

# Same data setup
micro_batch_size = 32
data = torch.randn(micro_batch_size, 1024, device=device)
targets = torch.randn(micro_batch_size, 512, device=device)

# ============================================================
# Training loop — hooks handle the all-reduce automatically
# ============================================================
num_steps = 5

for step in range(num_steps):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    
    # Forward pass — no change
    outputs = model(data)
    loss = nn.functional.mse_loss(outputs, targets)
    
    # Backward pass — hooks fire all-reduce per parameter automatically!
    # No manual all-reduce loop needed anymore.
    # As each parameter's gradient is computed, the hook fires
    # and NVLink starts transferring while CUDA cores keep computing.
    optimizer.zero_grad()
    loss.backward()
    
    # By the time backward finishes, most all-reduces are already done
    # (or close to done) because they overlapped with computation.
    
    # Optimizer step — gradients are already averaged thanks to hooks
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