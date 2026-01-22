import warnings
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic as nni
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.ao.quantization.fx.graph_module import _get_observed_graph_module_attr
from ..observer import _with_args, ObserverBase, PerChannelMinMaxObserver
from ..utils import _parent_name, check_min_max_valid
from .utils import (
def get_layer_sqnr_dict(model_a: nn.Module, model_b: nn.Module, x: torch.Tensor) -> Dict[str, float]:
    """ Runs the Numeric Suite on model_a and model_b and returns a dictionary
    containing the SQNR between layers in model_a and model_b.

    Note: In order to support equalized models, this function has a hacky fix in
    which we do not match any torch.mul operators. This is because equalized
    models contain extra mul operators to scale the input by the equalization
    scale, but this edge case has not been resolved yet within the numeric suite code.

    Args:
        model_a: A float model
        model_b: A quantized model
        x: Inputs to use during calibration
    """
    import torch.ao.ns._numeric_suite_fx as ns
    from torch.ao.ns.fx.mappings import get_unmatchable_types_map
    unmatchable_types_map = get_unmatchable_types_map()
    unmatchable_types_map['funs_unmatchable'].add(torch.mul)
    model_a_ns, model_b_ns = ns.add_loggers('fp32', model_a, 'int8', model_b, ns.OutputLogger, unmatchable_types_map=unmatchable_types_map)
    model_a_ns(x)
    model_b_ns(x)
    activation_comparison_dict = ns.extract_logger_info(model_a_ns, model_b_ns, ns.OutputLogger, 'int8')
    ns.extend_logger_results_with_comparison(activation_comparison_dict, 'fp32', 'int8', torch.ao.ns.fx.utils.compute_sqnr, 'sqnr')
    layer_sqnr_dict = {}
    for key in activation_comparison_dict:
        layer = activation_comparison_dict[key]['node_output']['int8'][0]['fqn']
        sqnr = activation_comparison_dict[key]['node_output']['int8'][0]['sqnr'][0]
        layer_sqnr_dict[layer] = sqnr
    return layer_sqnr_dict