import functools
import warnings
from typing import (
import torch
import optree
from optree import PyTreeSpec  # direct import for type annotations
def tree_map_(func: Callable[..., Any], tree: PyTree) -> PyTree:
    """Like :func:`tree_map`, but do an inplace call on each leaf and return the original tree.

    See also :func:`tree_map`.

    Args:
        func (callable): A function that takes a single argument, to be applied at the corresponding
            leaves of the pytree.
        tree (pytree): A pytree to be mapped over, with each leaf providing the argument to function
            ``func``.

    Returns:
        The original ``tree`` with the value at each leaf is given by the side-effect of function
        ``func(x)`` (not the return value) where ``x`` is the value at the corresponding leaf in
        ``tree``.
    """
    return optree.tree_map_(func, tree, none_is_leaf=True, namespace='torch')