from __future__ import annotations
import difflib
import functools
import itertools
import textwrap
from collections import OrderedDict, defaultdict, deque
from typing import Any, Callable, Iterable, Mapping, overload
from optree import _C
from optree.registry import (
from optree.typing import (
from optree.typing import structseq as PyStructSequence  # noqa: N812
from optree.typing import structseq_fields
def tree_map_with_path_(func: Callable[..., Any], tree: PyTree[T], *rests: PyTree[S], is_leaf: Callable[[T], bool] | None=None, none_is_leaf: bool=False, namespace: str='') -> PyTree[T]:
    """Like :func:`tree_map_with_path`, but do an inplace call on each leaf and return the original tree.

    See also :func:`tree_map`, :func:`tree_map_`, and :func:`tree_map_with_path`.

    Args:
        func (callable): A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees with extra paths.
        tree (pytree): A pytree to be mapped over, with each leaf providing the second positional
            argument and the corresponding path providing the first positional argument to function
            ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        The original ``tree`` with the value at each leaf is given by the side-effect of function
        ``func(p, x, *xs)`` (not the return value) where ``(p, x)`` are the path and value at the
        corresponding leaf in ``tree`` and ``xs`` is the tuple of values at values at corresponding
        nodes in ``rests``.
    """
    paths, leaves, treespec = _C.flatten_with_path(tree, is_leaf, none_is_leaf, namespace)
    flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
    deque(map(func, paths, *flat_args), maxlen=0)
    return tree