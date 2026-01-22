import functools
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
def maybe_convert_node(x: Any) -> Any:
    if not isinstance(x, torch.fx.Node):
        return x
    assert hasattr(shape_env, 'name_to_node')
    name_to_node = shape_env.name_to_node
    assert x.name in name_to_node
    return name_to_node[x.name]