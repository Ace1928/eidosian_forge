from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, Optional, Tuple, Type
import torch
import torch.fx
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
from .virtualized import V
def get_fake_args_kwargs(x: torch.fx.Node) -> Tuple[bool, Tuple[Any], Dict[str, Any]]:
    """
    First value returns a boolean if any of the input nodes don't have a faketensor.
    """
    args, kwargs = tree_map(get_fake, (x.args, x.kwargs))
    if any((isinstance(a, torch.fx.Node) for a in pytree.arg_tree_leaves(*args, **kwargs))):
        return (False, args, kwargs)
    return (True, args, kwargs)