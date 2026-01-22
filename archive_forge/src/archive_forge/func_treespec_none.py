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
def treespec_none(*, none_is_leaf: bool=False, namespace: str='') -> PyTreeSpec:
    """Make a treespec representing a :data:`None` node.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_tuple`.

    >>> treespec_none()
    PyTreeSpec(None)
    >>> treespec_none(none_is_leaf=True)
    PyTreeSpec(*, NoneIsLeaf)
    >>> treespec_none(none_is_leaf=False) == treespec_none(none_is_leaf=True)
    False
    >>> treespec_none() == tree_structure(None)
    True
    >>> treespec_none() == tree_structure(1)
    False
    >>> treespec_none(none_is_leaf=True) == tree_structure(1, none_is_leaf=True)
    True
    >>> treespec_none(none_is_leaf=True) == tree_structure(None, none_is_leaf=True)
    True
    >>> treespec_none(none_is_leaf=True) == tree_structure(None, none_is_leaf=False)
    False
    >>> treespec_none(none_is_leaf=True) == treespec_leaf(none_is_leaf=True)
    True
    >>> treespec_none(none_is_leaf=False) == treespec_leaf(none_is_leaf=True)
    False
    >>> treespec_none(none_is_leaf=True) == treespec_leaf(none_is_leaf=False)
    False
    >>> treespec_none(none_is_leaf=False) == treespec_leaf(none_is_leaf=False)
    False

    Args:
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a :data:`None` node.
    """
    return _C.make_none(none_is_leaf, namespace)