from typing import Optional, Union, ClassVar
from dataclasses import dataclass
import os
from packaging.version import Version
import torch
from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.utils import get_cpu_memory, is_hip, is_neuron, get_nvcc_cuda_version
def verify_with_model_config(self, model_config: ModelConfig):
    if self.lora_dtype in (None, 'auto'):
        self.lora_dtype = model_config.dtype
    elif isinstance(self.lora_dtype, str):
        self.lora_dtype = getattr(torch, self.lora_dtype)
    if model_config.quantization is not None:
        raise ValueError('LoRA is not supported with quantized models yet.')