import sys
import torch
from torch.fx.graph import (
from torch.ao.quantization.utils import Pattern
from .quantize_handler import (
from ..qconfig import (
from ..utils import (
from .graph_module import (
from torch.nn.utils.parametrize import type_before_parametrizations
from typing import Any, Dict, List, Callable, Optional, Tuple, Type, Set, Iterable

    Matches the nodes in the input graph to quantization patterns, and
    outputs the information needed to quantize them in future steps.

    Inputs:
      - graph: an fx.Graph object
      - modules: a mapping of fully qualified module name to instance,
          for example, {'foo': ModuleFoo, ...}
      - patterns: a mapping from a tuple of nodes in reverse order to
          uninitialized QuantizeHandler subclass.

    Outputs a map of
      node_name ->
        (node, matched_values, matched_pattern, QuantizeHandler instance,
         qconfig)

    For example, {
      'relu_1': (relu_1, [relu_1], torch.nn.functional.relu,
                 <CopyNodeQuantizeHandler instance>, QConfig(...)),
      ...
    }
    