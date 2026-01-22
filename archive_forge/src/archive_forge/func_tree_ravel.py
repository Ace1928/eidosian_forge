from __future__ import annotations
import itertools
import warnings
from types import FunctionType
from typing import Any, Callable
from typing_extensions import TypeAlias  # Python 3.10+
import jax.numpy as jnp
from jax import Array, lax
from jax._src import dtypes
from jax.typing import ArrayLike
from optree.ops import tree_flatten, tree_unflatten
from optree.typing import PyTreeSpec, PyTreeTypeVar
from optree.utils import safe_zip, total_order_sorted
def tree_ravel(tree: ArrayLikeTree, is_leaf: Callable[[Any], bool] | None=None, *, none_is_leaf: bool=False, namespace: str='') -> tuple[Array, Callable[[Array], ArrayTree]]:
    """Ravel (flatten) a pytree of arrays down to a 1D array.

    >>> tree = {
    ...     'layer1': {
    ...         'weight': jnp.arange(0, 6, dtype=jnp.float32).reshape((2, 3)),
    ...         'bias': jnp.arange(6, 8, dtype=jnp.float32).reshape((2,)),
    ...     },
    ...     'layer2': {
    ...         'weight': jnp.arange(8, 10, dtype=jnp.float32).reshape((1, 2)),
    ...         'bias': jnp.arange(10, 11, dtype=jnp.float32).reshape((1,))
    ...     },
    ... }
    >>> tree
    {'layer1': {'weight': Array([[0., 1., 2.],
                                 [3., 4., 5.]], dtype=float32),
                'bias': Array([6., 7.], dtype=float32)},
     'layer2': {'weight': Array([[8., 9.]], dtype=float32),
                'bias': Array([10.], dtype=float32)}}
    >>> flat, unravel_func = tree_ravel(tree)
    >>> flat
    Array([ 6.,  7.,  0.,  1.,  2.,  3.,  4.,  5., 10.,  8.,  9.], dtype=float32)
    >>> unravel_func(flat)
    {'layer1': {'weight': Array([[0., 1., 2.],
                                 [3., 4., 5.]], dtype=float32),
                'bias': Array([6., 7.], dtype=float32)},
     'layer2': {'weight': Array([[8., 9.]], dtype=float32),
                'bias': Array([10.], dtype=float32)}}

    Args:
        tree (pytree): a pytree of arrays and scalars to ravel.
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
        A pair ``(array, unravel_func)`` where the first element is a 1D array representing the
        flattened and concatenated leaf values, with ``dtype`` determined by promoting the
        ``dtype``\\s of leaf values, and the second element is a callable for unflattening a 1D array
        of the same length back to a pytree of the same structure as the input ``tree``. If the
        input pytree is empty (i.e. has no leaves) then as a convention a 1D empty array of the
        default dtype is returned in the first component of the output.
    """
    leaves, treespec = tree_flatten(tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
    flat, unravel_flat = _ravel_leaves(leaves)
    return (flat, HashablePartial(_tree_unravel, treespec, unravel_flat))