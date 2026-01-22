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
def tree_sum(tree: PyTree[T], start: T=0, *, is_leaf: Callable[[T], bool] | None=None, none_is_leaf: bool=False, namespace: str='') -> T:
    """Sum ``start`` and leaf values in ``tree`` in left-to-right depth-first order and return the total.

    See also :func:`tree_leaves` and :func:`tree_reduce`.

    >>> tree_sum({'x': 1, 'y': (2, 3)})
    6
    >>> tree_sum({'x': 1, 'y': (2, None), 'z': 3})  # `None` is a non-leaf node with arity 0 by default
    6
    >>> tree_sum({'x': 1, 'y': (2, None), 'z': 3}, none_is_leaf=True)
    Traceback (most recent call last):
        ...
    TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'
    >>> tree_sum({'x': 'a', 'y': ('b', None), 'z': 'c'}, start='')
    'abc'
    >>> tree_sum({'x': [1], 'y': ([2], [None]), 'z': [3]}, start=[], is_leaf=lambda x: isinstance(x, list))
    [1, 2, None, 3]

    Args:
        tree (pytree): A pytree to be traversed.
        start (object, optional): An initial value to be used for the sum. (default: :data:`0`)
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
        The total sum of ``start`` and leaf values in ``tree``.
    """
    leaves = tree_leaves(tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
    if isinstance(start, str):
        return ''.join([start, *leaves])
    if isinstance(start, (bytes, bytearray)):
        return b''.join([start, *leaves])
    return sum(leaves, start)