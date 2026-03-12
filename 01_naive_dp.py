"""
Step 1: Naive Data Parallelism
===============================
The simplest possible version:
- Each GPU gets a full copy of the model
- Each GPU processes a different micro-batch
- After backward pass, all-reduce gradients across all GPUs
- Optimizer step with identical gradients = identical models

Run with:
  torchrun --nproc_per_node=8 01_naive_dp.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW

try:
  destroyed = False
  # ============================================================
  # 1. Initialize distributed — every GPU runs this same script
  # ============================================================
  dist.init_process_group(backend="nccl")

  rank = dist.get_rank()           # which GPU am I? (0-7)
  world_size = dist.get_world_size()  # how many GPUs total? (8)
  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)

  if rank == 0:
      print(f"World size: {world_size}")
      print("=" * 60)

  # ============================================================
  # 2. Create a simple model — same on every GPU
  # ============================================================
  # We use a small model so we can focus on the parallelism pattern
  # In real training this would be a transformer with billions of params

  torch.manual_seed(42)  # same seed = same initial weights on all GPUs

  model = nn.Sequential(
      nn.Linear(1024, 2048),
      nn.ReLU(),
      nn.Linear(2048, 2048),
      nn.ReLU(),
      nn.Linear(2048, 512),
  ).to(device)

  num_params = sum(p.numel() for p in model.parameters())
  if rank == 0:
      print(f"Model parameters: {num_params:,}")
      print(f"Model memory (BF16): {num_params * 2 / 1e6:.1f} MB")
      print(f"Optimizer memory (Adam): {num_params * 12 / 1e6:.1f} MB")
      print(f"Total per GPU: {num_params * 16 / 1e6:.1f} MB")
      print("=" * 60)

  optimizer = AdamW(model.parameters(), lr=1e-3)

  # ============================================================
  # 3. Create different data for each GPU
  # ============================================================
  # This is the "data parallelism" part — each GPU sees different data
  # In real training, a DataLoader would handle this split

  micro_batch_size = 32
  # Each GPU gets its own random data (simulating different micro-batches)
  data = torch.randn(micro_batch_size, 1024, device=device)
  targets = torch.randn(micro_batch_size, 512, device=device)

  if rank == 0:
      print(f"Micro-batch size per GPU: {micro_batch_size}")
      print(f"Global batch size: {micro_batch_size * world_size}")
      print("=" * 60)

  # ============================================================
  # 4. Training loop — the naive approach
  # ============================================================
  num_steps = 5

  for step in range(num_steps):
      torch.cuda.synchronize()
      start = torch.cuda.Event(enable_timing=True)
      end = torch.cuda.Event(enable_timing=True)
      
      start.record()
      
      # ----- Forward pass -----
      # Each GPU runs forward on its own micro-batch
      # No communication needed here
      outputs = model(data)
      loss = nn.functional.mse_loss(outputs, targets)
      
      # ----- Backward pass -----
      # Each GPU computes gradients on its own data
      # No communication yet — gradients are LOCAL and DIFFERENT on each GPU
      optimizer.zero_grad()
      loss.backward()
      
      # ----- THE NAIVE PART: All-reduce gradients -----
      # This is where we sync. GPU sits IDLE during this communication.
      # We manually all-reduce every parameter's gradient.
      # In the naive approach, this happens AFTER the full backward pass.
      for param in model.parameters():
          if param.grad is not None:
              # Sum gradients across all GPUs, then divide by world_size to average
              dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
              param.grad /= world_size
      
      # ----- Optimizer step -----
      # Now every GPU has identical averaged gradients
      # So Adam will produce identical weight updates
      # Models stay in sync!
      optimizer.step()
      
      end.record()
      torch.cuda.synchronize()
      
      step_time = start.elapsed_time(end)  # milliseconds
      
      if rank == 0:
          print(f"Step {step}: loss={loss.item():.4f}, time={step_time:.2f}ms")

  # ============================================================
  # 5. Verify all GPUs have identical weights
  # ============================================================
  # This confirms our all-reduce kept models in sync
  with torch.no_grad():
      first_param = next(model.parameters())
      param_sum = first_param.sum().clone()
      
      # Gather sums from all GPUs
      all_sums = [torch.zeros_like(param_sum) for _ in range(world_size)]
      dist.all_gather(all_sums, param_sum)
      
      if rank == 0:
          all_same = all(torch.allclose(all_sums[0], s) for s in all_sums)
          print("=" * 60)
          print(f"All GPUs have identical weights: {all_same}")

  dist.destroy_process_group()
  destroyed = True
except Exception as e:
  
  dist.destroy_process_group()
finally:
  if not destroyed:
    dist.destroy_process_group()