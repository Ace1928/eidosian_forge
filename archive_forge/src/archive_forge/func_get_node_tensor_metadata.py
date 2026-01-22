import logging
import os
import tempfile
from enum import Enum
from typing import Callable, cast, Dict, Iterable, List, Set
import torch.fx as fx
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
def get_node_tensor_metadata(node: fx.Node, is_required: bool=True) -> TensorMetadata:
    metadata = node.meta.get('tensor_meta', None)
    if is_required and metadata is None:
        raise RuntimeError(f'Callsite expects that ``tensor_meta`` exists in ``{node.name}``, but got None instead. Node: {node.op} {node.name} {node.target}')
    return metadata