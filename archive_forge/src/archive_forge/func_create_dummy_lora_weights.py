from typing import List, Optional
import torch
from vllm.utils import in_wsl
@classmethod
def create_dummy_lora_weights(cls, module_name: str, input_dim: int, output_dim: int, rank: int, dtype: torch.dtype, device: torch.device, embeddings_tensor_dim: Optional[int]=None) -> 'LoRALayerWeights':
    pin_memory = str(device) == 'cpu' and (not in_wsl())
    lora_a = torch.zeros([input_dim, rank], dtype=dtype, device=device, pin_memory=pin_memory)
    lora_b = torch.zeros([rank, output_dim], dtype=dtype, device=device, pin_memory=pin_memory)
    embeddings_tensor = torch.rand(10, embeddings_tensor_dim, dtype=dtype, device=device, pin_memory=pin_memory) if embeddings_tensor_dim else None
    return cls(module_name, rank=rank, lora_alpha=1, lora_a=lora_a, lora_b=lora_b, embeddings_tensor=embeddings_tensor)