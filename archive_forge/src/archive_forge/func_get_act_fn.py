import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm._C import ops
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils.utils import divide
from vllm.model_executor.utils import set_weight_attrs
def get_act_fn(act_fn_name: str, quant_config: Optional[QuantizationConfig]=None, intermediate_size: Optional[int]=None, input_is_parallel: bool=True, params_dtype: Optional[torch.dtype]=None) -> nn.Module:
    """Get an activation function by name."""
    act_fn_name = act_fn_name.lower()
    if act_fn_name not in _ACTIVATION_REGISTRY:
        raise ValueError(f'Activation function {act_fn_name!r} is not supported.')
    act_fn = _ACTIVATION_REGISTRY[act_fn_name]
    if quant_config is not None and act_fn_name in quant_config.get_scaled_act_names():
        if intermediate_size is None:
            raise ValueError('intermediate_size must be specified for scaled activation functions.')
        return ScaledActivation(act_fn, intermediate_size, input_is_parallel, params_dtype)
    return act_fn