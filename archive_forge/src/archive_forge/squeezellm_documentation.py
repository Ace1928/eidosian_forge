from typing import Any, Dict, List, Optional
import torch
from torch.nn.parameter import Parameter
from vllm._C import ops
from vllm.model_executor.layers.linear import (LinearMethodBase,
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.utils import is_hip
Linear method for SqueezeLLM.

    Args:
        quant_config: The SqueezeLLM quantization config.
    