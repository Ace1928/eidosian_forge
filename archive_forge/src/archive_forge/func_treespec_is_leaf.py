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
def treespec_is_leaf(treespec: PyTreeSpec, strict: bool=True) -> bool:
    """Return whether the treespec is a leaf that has no children.

    See also :func:`treespec_is_strict_leaf` and :meth:`PyTreeSpec.is_leaf`.

    This function is equivalent to ``treespec.is_leaf(strict=strict)``. If ``strict=False``, it will
    return :data:`True` if and only if the treespec represents a strict leaf. If ``strict=False``,
    it will return :data:`True` if the treespec represents a strict leaf or :data:`None` or an empty
    container (e.g., an empty tuple).

    >>> treespec_is_leaf(tree_structure(1))
    True
    >>> treespec_is_leaf(tree_structure((1, 2)))
    False
    >>> treespec_is_leaf(tree_structure(None))
    False
    >>> treespec_is_leaf(tree_structure(None), strict=False)
    True
    >>> treespec_is_leaf(tree_structure(None, none_is_leaf=False))
    False
    >>> treespec_is_leaf(tree_structure(None, none_is_leaf=True))
    True
    >>> treespec_is_leaf(tree_structure(()))
    False
    >>> treespec_is_leaf(tree_structure(()), strict=False)
    True
    >>> treespec_is_leaf(tree_structure([]))
    False
    >>> treespec_is_leaf(tree_structure([]), strict=False)
    True

    Args:
        treespec (PyTreeSpec): A treespec.
        strict (bool, optional): Whether not to treat :data:`None` or an empty
            container (e.g., an empty tuple) as a leaf. (default: :data:`True`)

    Returns:
        :data:`True` if the treespec represents a leaf that has no children, otherwise, :data:`False`.
    """
    if strict:
        return treespec.num_nodes == 1 and treespec.num_leaves == 1
    return treespec.num_nodes == 1