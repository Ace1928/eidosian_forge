import importlib
from typing import TYPE_CHECKING, Any, Dict, List, Union
from packaging import version
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name
from ..utils import is_accelerate_available, is_bitsandbytes_available, is_torch_available, logging
def check_quantized_param(self, model: 'PreTrainedModel', param_value: 'torch.Tensor', param_name: str, state_dict: Dict[str, Any]) -> bool:
    import bitsandbytes as bnb
    module, tensor_name = get_module_from_name(model, param_name)
    if isinstance(module._parameters[tensor_name], bnb.nn.Params4bit):
        return True
    elif isinstance(module, bnb.nn.Linear4bit) and tensor_name == 'bias':
        return True
    else:
        return False