import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union, overload
from ._functions import Scatter, Gather
import warnings
def scatter_map(obj):
    if isinstance(obj, torch.Tensor):
        return Scatter.apply(target_gpus, None, dim, obj)
    if _is_namedtuple(obj):
        return [type(obj)(*args) for args in zip(*map(scatter_map, obj))]
    if isinstance(obj, tuple) and len(obj) > 0:
        return list(zip(*map(scatter_map, obj)))
    if isinstance(obj, list) and len(obj) > 0:
        return [list(i) for i in zip(*map(scatter_map, obj))]
    if isinstance(obj, dict) and len(obj) > 0:
        return [type(obj)(i) for i in zip(*map(scatter_map, obj.items()))]
    return [obj for _ in target_gpus]