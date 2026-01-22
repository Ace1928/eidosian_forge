import logging
import os
import tempfile
from enum import Enum
from typing import Callable, cast, Dict, Iterable, List, Set
import torch.fx as fx
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
def rebuild_graph(gm: fx.GraphModule, remove_dead_code: bool=True) -> None:
    """Run the required steps to ensure production-ready graph.

    Note - per the fx docs, elimination of dead code is not very precise.
    Hence, the flag to make this step optional.
    """
    gm.graph.lint()
    if remove_dead_code:
        gm.graph.eliminate_dead_code()
    gm.recompile()