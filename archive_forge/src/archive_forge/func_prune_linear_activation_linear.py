from typing import cast, List, Optional, Callable, Tuple
import torch
from torch import nn, Tensor
from torch.nn.utils import parametrize
from torch.nn.utils.parametrize import ParametrizationList
from .parametrization import FakeStructuredSparsity, BiasHook
def prune_linear_activation_linear(linear1: nn.Linear, activation: Optional[Callable[[Tensor], Tensor]], linear2: nn.Linear):
    mask = _prune_linear_helper(linear1)
    if getattr(linear1, 'prune_bias', False):
        _prune_module_bias(linear1, mask)
    else:
        pruned_biases = _propogate_module_bias(linear1, mask)
        if pruned_biases is not None:
            if activation:
                pruned_biases = activation(pruned_biases)
            linear2.bias = _get_adjusted_next_layer_bias(linear2, pruned_biases, mask)
    with torch.no_grad():
        if parametrize.is_parametrized(linear2):
            parametrization_dict = cast(nn.ModuleDict, linear2.parametrizations)
            weight_parameterizations = cast(ParametrizationList, parametrization_dict.weight)
            weight_parameterizations.original = nn.Parameter(weight_parameterizations.original[:, mask])
            linear2.in_features = weight_parameterizations.original.shape[1]
        else:
            linear2.weight = nn.Parameter(linear2.weight[:, mask])
            linear2.in_features = linear2.weight.shape[1]