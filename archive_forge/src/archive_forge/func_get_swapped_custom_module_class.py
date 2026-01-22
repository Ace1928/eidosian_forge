import functools
import warnings
from collections import OrderedDict
from inspect import getfullargspec, signature
from typing import Any, Callable, Dict, Optional, Tuple, Union
import torch
from torch.ao.quantization.quant_type import QuantType
from torch.fx import Node
from torch.nn.utils.parametrize import is_parametrized
def get_swapped_custom_module_class(custom_module, custom_module_class_mapping, qconfig):
    """ Get the observed/quantized custom module class that we need
    to swap `custom_module` to
    Input:
        custom_module: input, can be an instance of either a float or observed custom module
        custom_module_class_mapping: the float to observed or observed to quantized custom module class mapping
        qconfig: qconfig configured for the custom module

    Output:
        corresponding observed/quantized custom module class for input custom module instance
    """
    quant_type = get_quant_type(qconfig)
    class_mapping = custom_module_class_mapping.get(quant_type, {})
    assert type(custom_module) in class_mapping, f'did not find corresponding observed module class for {type(custom_module)} in mapping: {class_mapping}'
    return class_mapping[type(custom_module)]