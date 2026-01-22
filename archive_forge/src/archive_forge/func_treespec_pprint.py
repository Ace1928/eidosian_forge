import functools
import warnings
from typing import (
import torch
import optree
from optree import PyTreeSpec  # direct import for type annotations
def treespec_pprint(treespec: TreeSpec) -> str:
    dummy_tree = tree_unflatten([_DummyLeaf() for _ in range(treespec.num_leaves)], treespec)
    return repr(dummy_tree)