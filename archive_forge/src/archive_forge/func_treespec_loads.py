import functools
import warnings
from typing import (
import torch
import optree
from optree import PyTreeSpec  # direct import for type annotations
def treespec_loads(serialized: str) -> TreeSpec:
    """Deserialize a treespec from a JSON string."""
    from ._pytree import tree_unflatten as _tree_unflatten, treespec_loads as _treespec_loads
    orig_treespec = _treespec_loads(serialized)
    dummy_tree = _tree_unflatten([0] * orig_treespec.num_leaves, orig_treespec)
    treespec = tree_structure(dummy_tree)
    return treespec