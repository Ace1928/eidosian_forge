from typing import List, Optional
import torch
from vllm.lora.lora import LoRALayerWeights, PackedLoRALayerWeights
def init_random_lora(self, module_name: str, weight: torch.Tensor, rank: int=8, generate_embeddings_tensor: int=0):
    lora = LoRALayerWeights(module_name, rank=rank, lora_alpha=1, lora_a=torch.rand([weight.shape[1], rank], dtype=weight.dtype, device='cuda'), lora_b=torch.rand([rank, weight.shape[0]], dtype=weight.dtype, device='cuda'))
    if generate_embeddings_tensor:
        lora.embeddings_tensor = torch.rand(5, generate_embeddings_tensor, dtype=weight.dtype, device='cuda')
    self.set_module_lora(module_name, lora)
    return lora