from functools import partial
from typing import Any, Callable, Dict, List, Type, TypeVar, Union, overload
import torch
import torch.nn as nn
import torch.types
def tree_map(fn, tree, leaf_type):
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple((tree_map(fn, x, leaf_type) for x in tree))
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        print(type(tree))
        raise ValueError('Not supported')