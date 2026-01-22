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
@pytest.fixture
def dummy_model_gate_up() -> nn.Module:
    model = nn.Sequential(OrderedDict([('dense1', ColumnParallelLinear(764, 100)), ('dense2', RowParallelLinear(100, 50)), ('layer1', nn.Sequential(OrderedDict([('dense1', ColumnParallelLinear(100, 10)), ('dense2', RowParallelLinear(10, 50))]))), ('act2', nn.ReLU()), ('gate_up_proj', MergedColumnParallelLinear(50, [5, 5])), ('outact', nn.Sigmoid()), ('lm_head', ParallelLMHead(512, 10)), ('sampler', Sampler(512))]))
    model.config = MagicMock()
    return model