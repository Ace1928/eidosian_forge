from .graph_module import GraphModule
from .graph import Graph
from .node import Node
from ._symbolic_trace import symbolic_trace
from ._compatibility import compatibility
import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Union
import torch
def try_get_attr(gm: torch.nn.Module, target: str) -> Optional[Any]:
    module_path, _, attr_name = target.rpartition('.')
    mod: torch.nn.Module = gm.get_submodule(module_path)
    attr = getattr(mod, attr_name, None)
    return attr