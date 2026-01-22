from typing import Optional, Union, ClassVar
from dataclasses import dataclass
import os
from packaging.version import Version
import torch
from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.utils import get_cpu_memory, is_hip, is_neuron, get_nvcc_cuda_version
def get_num_kv_heads(self, parallel_config: 'ParallelConfig') -> int:
    """Returns the number of KV heads per GPU."""
    total_num_kv_heads = self.get_total_num_kv_heads()
    return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)