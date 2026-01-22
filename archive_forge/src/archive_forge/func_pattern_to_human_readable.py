from typing import Dict, Any, List, Callable, Union, Tuple, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backend_config import (
from ..utils import Pattern
from ..fuser_method_mappings import (
def pattern_to_human_readable(p) -> Any:
    if isinstance(p, tuple):
        return tuple((pattern_to_human_readable(inner_p) for inner_p in p))
    elif isinstance(p, str):
        return p
    else:
        p = remove_boolean_dispatch_from_name(p)
        return p