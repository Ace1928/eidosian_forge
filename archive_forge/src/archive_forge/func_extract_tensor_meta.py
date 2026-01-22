import torch
import torch.fx
import traceback
from torch._dispatch.python import enable_python_dispatcher
from torch.fx.node import Node, map_aggregate
from typing import Any, Tuple, NamedTuple, Optional, Dict
from torch.fx._compatibility import compatibility
from torch._guards import detect_fake_mode
def extract_tensor_meta(obj):
    if isinstance(obj, torch.Tensor):
        nonlocal found_tensor
        found_tensor = True
        return _extract_tensor_metadata(obj)
    else:
        return obj