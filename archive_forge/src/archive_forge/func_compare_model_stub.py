import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
from torch.ao.quantization import prepare
from typing import Dict, List, Optional, Any, Union, Callable, Set
from torch.ao.quantization.quantization_mappings import (
def compare_model_stub(float_model: nn.Module, q_model: nn.Module, module_swap_list: Set[type], *data, logger_cls=ShadowLogger) -> Dict[str, Dict]:
    """Compare quantized module in a model with its floating point counterpart,
    feeding both of them the same input. Return a dict with key corresponding to
    module names and each entry being a dictionary with two keys 'float' and
    'quantized', containing the output tensors of quantized and its matching
    float shadow module. This dict can be used to compare and compute the module
    level quantization error.

    This function first call prepare_model_with_stubs() to swap the quantized
    module that we want to compare with the Shadow module, which takes quantized
    module, corresponding float module and logger as input, and creates a forward
    path inside to make the float module to shadow quantized module sharing the
    same input. The logger can be customizable, default logger is ShadowLogger
    and it will save the outputs of the quantized module and float module that
    can be used to compute the module level quantization error.

    Example usage::

        module_swap_list = [torchvision.models.quantization.resnet.QuantizableBasicBlock]
        ob_dict = compare_model_stub(float_model,qmodel,module_swap_list, data)
        for key in ob_dict:
            print(key, compute_error(ob_dict[key]['float'], ob_dict[key]['quantized'].dequantize()))

    Args:
        float_model: float model used to generate the q_model
        q_model: model quantized from float_model
        module_swap_list: list of float module types at which shadow modules will
            be attached.
        data: input data used to run the prepared q_model
        logger_cls: type of logger to be used in shadow module to process the outputs of
            quantized module and its float shadow module
    """
    torch._C._log_api_usage_once('quantization_api._numeric_suite.compare_model_stub')
    prepare_model_with_stubs(float_model, q_model, module_swap_list, logger_cls)
    q_model(*data)
    ob_dict = get_logger_dict(q_model)
    return ob_dict