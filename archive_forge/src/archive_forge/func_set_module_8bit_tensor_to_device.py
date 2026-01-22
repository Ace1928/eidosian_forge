import importlib.metadata
import warnings
from copy import deepcopy
from packaging import version
from ..utils import is_accelerate_available, is_bitsandbytes_available, logging
def set_module_8bit_tensor_to_device(*args, **kwargs):
    warnings.warn('`set_module_8bit_tensor_to_device` will be deprecated in a future version, please use `set_module_quantized_tensor_to_device` instead', FutureWarning)
    return set_module_quantized_tensor_to_device(*args, **kwargs)