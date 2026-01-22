import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.fx
from torch.fx._compatibility import compatibility
from torch.fx.node import map_arg
from .shape_prop import ShapeProp
from .split_utils import split_by_tags
from .tools_common import (
def run_b(self, mod: torch.fx.GraphModule, inputs: Tensors) -> TensorOrTensors:
    """
        Run `mod` with `inputs` and generate output. The output will be compared with
        output of run_a().
        """
    raise RuntimeError('run_b() is not implemented.')