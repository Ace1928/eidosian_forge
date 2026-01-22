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
def update_obs_for_equalization(model: GraphModule, modules: Dict[str, nn.Module]) -> Dict[str, _WeightEqualizationObserver]:
    """ Update all of the observer's equalization scale. For each
    InputEqualizationObserver, we will find the location of the next
    WeightEqualizationObserver, create it, and calculate the equalization scale
    based on the two observers.

    We will then return a dictionary mapping operation node names to
    the corresponding WeightEqualizationObservers for that operation.
    """
    weight_eq_obs_dict = {}
    for node in model.graph.nodes:
        if node.op == 'call_module' and isinstance(modules[node.target], _InputEqualizationObserver):
            input_eq_obs = modules[node.target]
            assert isinstance(input_eq_obs, _InputEqualizationObserver)
            op_node, weight_eq_obs = get_op_node_and_weight_eq_obs(node, model, modules)
            if op_node is None or weight_eq_obs is None:
                continue
            if op_node.op == 'call_module':
                if fused_module_supports_equalization(modules[str(op_node.target)]):
                    module = modules[str(op_node.target)][0]
                    assert nn_module_supports_equalization(module)
                    weight_eq_obs(module.weight)
                else:
                    weight_eq_obs(modules[str(op_node.target)].weight)
            equalization_scale = calculate_equalization_scale(input_eq_obs, weight_eq_obs)
            input_eq_obs.set_equalization_scale(equalization_scale)
            weight_eq_obs.set_equalization_scale(equalization_scale)
            weight_eq_obs_dict[op_node.name] = weight_eq_obs
    return weight_eq_obs_dict