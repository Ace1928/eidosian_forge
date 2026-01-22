import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def tree_iter(tree, is_leaf=is_not_container):
    """Iterate over all leaves in ``tree``.

    Parameters
    ----------
    f : callable
        A function to apply to all leaves in ``tree``.
    tree : pytree
        A nested sequence of tuples, lists, dicts and other objects.
    is_leaf : callable
        A function to determine if an object is a leaf, ``f`` is only applied
        to objects for which ``is_leaf(x)`` returns ``True``.
    """
    if is_leaf(tree):
        yield tree
        return
    try:
        yield from TREE_ITER_CACHE[tree.__class__](tree, is_leaf)
    except KeyError:
        for cls, iterator in reversed(TREE_ITER_REGISTRY.items()):
            if isinstance(tree, cls):
                break
        else:
            iterator = empty
        TREE_ITER_CACHE[tree.__class__] = iterator
        yield from iterator(tree, is_leaf)