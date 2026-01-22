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
def tree_is_leaf(tree: PyTree[T], is_leaf: Callable[[T], bool] | None=None, *, none_is_leaf: bool=False, namespace: str='') -> bool:
    """Test whether the given object is a leaf node.

    See also :func:`tree_flatten`, :func:`tree_leaves`, and :func:`all_leaves`.

    >>> tree_is_leaf(1)
    True
    >>> tree_is_leaf(None)
    False
    >>> tree_is_leaf(None, none_is_leaf=True)
    True
    >>> tree_is_leaf({'a': 1, 'b': (2, 3)})
    False

    Args:
        tree (pytree): A pytree to check if it is a leaf node.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than a leaf. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A boolean indicating if all elements in the input iterable are leaves.
    """
    return _C.is_leaf(tree, is_leaf, none_is_leaf, namespace)