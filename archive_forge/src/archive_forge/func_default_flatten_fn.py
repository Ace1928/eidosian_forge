import dataclasses
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
import torch
from torch._export import ExportedProgram
from torch.utils._pytree import (
def default_flatten_fn(obj: Any) -> Tuple[List[Any], Context]:
    flattened = []
    flat_names = []
    none_names = []
    for f in dataclasses.fields(obj):
        name, val = (f.name, getattr(obj, f.name))
        if val is not None or return_none_fields:
            flattened.append(val)
            flat_names.append(name)
        else:
            none_names.append(name)
    return (flattened, (cls, flat_names, none_names))