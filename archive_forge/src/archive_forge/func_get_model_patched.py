import contextlib
import gc
import tempfile
from collections import OrderedDict
from unittest.mock import patch, MagicMock
import pytest
import ray
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
import vllm
from vllm.config import LoRAConfig
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.parallel_utils.parallel_state import (
def get_model_patched(model_config, device_config, **kwargs):
    return get_model_old(model_config, device_config, lora_config=LoRAConfig(max_loras=4, max_lora_rank=8))