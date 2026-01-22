import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
from torch.ao.quantization import prepare
from typing import Dict, List, Optional, Any, Union, Callable, Set
from torch.ao.quantization.quantization_mappings import (
def prepare_model_outputs(float_module: nn.Module, q_module: nn.Module, logger_cls=OutputLogger, allow_list=None) -> None:
    """Prepare the model by attaching the logger to both float module
    and quantized module if they are in the allow_list.

    Args:
        float_module: float module used to generate the q_module
        q_module: module quantized from float_module
        logger_cls: type of logger to be attached to float_module and q_module
        allow_list: list of module types to attach logger
    """
    torch._C._log_api_usage_once('quantization_api._numeric_suite.prepare_model_outputs')
    if allow_list is None:
        allow_list = get_default_compare_output_module_list()
    qconfig_debug = torch.ao.quantization.QConfig(activation=logger_cls, weight=None)
    float_module.qconfig = qconfig_debug
    prepare(float_module, inplace=True, allow_list=allow_list, prepare_custom_config_dict={})
    q_module.qconfig = qconfig_debug
    prepare(q_module, inplace=True, allow_list=allow_list, observer_non_leaf_module_list=NON_LEAF_MODULE_TO_ADD_OBSERVER_ALLOW_LIST, prepare_custom_config_dict={})