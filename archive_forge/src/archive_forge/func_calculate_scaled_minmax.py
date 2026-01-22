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
def calculate_scaled_minmax(self):
    """ Returns the scaled min/max inputs
        """
    if self.equalization_scale.nelement() == 1 and self.equalization_scale == torch.tensor(1):
        warnings.warn('Must call calculate_equalization_scale before calling calculate_scaled_minmax. ' + 'Will not scale the next quantization observer.')
        return (None, None)
    min_inputs, max_inputs = self.get_input_minmax()
    equalization_scale_reshaped = reshape_scale(self.equalization_scale, 0, min_inputs)
    min_input_scaled = torch.min(torch.mul(min_inputs, equalization_scale_reshaped))
    max_input_scaled = torch.max(torch.mul(max_inputs, equalization_scale_reshaped))
    return (min_input_scaled, max_input_scaled)