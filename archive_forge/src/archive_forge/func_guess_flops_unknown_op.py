import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Set, Tuple
import torch.cuda.memory
import torch.cuda.nvtx
import torch.profiler
import torch.utils.hooks
from torch.utils._python_dispatch import TorchDispatchMode, _pop_mode_temporarily
from torch.utils._pytree import tree_map
from ..ops.common import FUNC_TO_XFORMERS_OPERATOR
from .device_limits import get_device_limits
from .profiler import _Profiler
def guess_flops_unknown_op(inputs: List[Any], outputs: List[Any]):
    total_elements = 0

    def process(x) -> None:
        nonlocal total_elements
        if isinstance(x, torch.Tensor):
            total_elements += x.numel()
    tree_map(process, inputs)
    tree_map(process, outputs)
    return total_elements / 2