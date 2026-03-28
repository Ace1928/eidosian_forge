from __future__ import annotations

import torch
import torch.nn as torch_nn
from .layers import BitLinear

class HyperExpert(torch_nn.Module):
    """
    A dynamic expert whose weights are generated on-the-fly by a Hypernetwork.
    """
    def __init__(self, input_dim: int, output_dim: int, hyper_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # The Hypernetwork: generates weights for a BitLinear-like operation
        self.generator = torch_nn.Sequential(
            torch_nn.Linear(input_dim, hyper_dim),
            torch_nn.ReLU(),
            torch_nn.Linear(hyper_dim, input_dim * output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, L, D)
        B, L, D = x.shape
        
        # Generate weights based on the mean input context
        context = x.mean(dim=1) # (B, D)
        weights = self.generator(context) # (B, D*out)
        weights = weights.view(B, self.output_dim, self.input_dim)
        
        # Perform the dynamic linear operation: out = x @ W.T
        # Using batch matrix multiplication
        out = torch.bmm(x, weights.transpose(1, 2)) # (B, L, out)
        return out

class BitMoE(torch_nn.Module):
    """
    Eidosian Mixture of Experts with Dynamic Expert Synthesis and Inhibitory Consensus.
    """
    def __init__(self, dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 1. Static Experts (Initial specialized knowledge)
        self.experts = torch_nn.ModuleList([
            torch_nn.Sequential(BitLinear(dim, dim * 4), torch_nn.GELU(), BitLinear(dim * 4, dim))
            for _ in range(num_experts)
        ])
        
        # 2. Hypernetwork for Dynamic Expert Generation
        self.hyper_expert = HyperExpert(dim, dim)
        
        # 3. Router
        self.router = torch_nn.Linear(dim, num_experts + 1) # +1 for the hyper-expert
        
        # 4. Inhibitory Consensus System
        self.inhibitor = torch_nn.Sequential(
            torch_nn.Linear(dim * (top_k), dim),
            torch_nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        # Router scores
        logits = self.router(x.mean(dim=1)) # (B, num_experts + 1)
        probs = torch.softmax(logits, dim=-1)
        
        # Select top-k experts
        top_probs, top_indices = torch.topk(probs, self.top_k, dim=-1)
        
        expert_outputs = []
        for i in range(self.top_k):
            # Check if we selected a static expert or the hyper-expert
            expert_idx = top_indices[:, i] # (B,)
            
            # This is a simplified vectorized version for the prototype
            # In production, we'd use a more efficient sparse routing kernel
            batch_out = torch.zeros_like(x)
            for b in range(B):
                idx = expert_idx[b].item()
                if idx < self.num_experts:
                    batch_out[b] = self.experts[idx](x[b:b+1])
                else:
                    batch_out[b] = self.hyper_expert(x[b:b+1])
            expert_outputs.append(batch_out)
            
        # Combine expert outputs
        combined = torch.cat(expert_outputs, dim=-1) # (B, L, D*top_k)
        
        # Inhibitory Gating: Consensus layer
        gate = self.inhibitor(combined.mean(dim=1)).unsqueeze(1) # (B, 1, D)
        
        # Weighted sum of expert outputs
        final_out = sum(expert_outputs[i] * top_probs[:, i].view(B, 1, 1) for i in range(self.top_k))
        
        # Apply the inhibitory gate
        return final_out * gate
