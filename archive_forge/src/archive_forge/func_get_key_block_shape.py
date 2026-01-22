from typing import Dict, List, Tuple
import torch
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import in_wsl, is_neuron, STR_DTYPE_TO_TORCH_DTYPE
def get_key_block_shape(self) -> Tuple[int, int, int, int]:
    element_size = torch.tensor([], dtype=self.dtype).element_size()
    x = 16 // element_size
    return (self.num_heads, self.head_size // x, self.block_size, x)