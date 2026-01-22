from collections.abc import Sequence, Hashable
from numbers import Integral
from functools import reduce
from typing import Generic, TypeVar
def plist(iterable=(), reverse=False):
    """
    Creates a new persistent list containing all elements of iterable.
    Optional parameter reverse specifies if the elements should be inserted in
    reverse order or not.

    >>> plist([1, 2, 3])
    plist([1, 2, 3])
    >>> plist([1, 2, 3], reverse=True)
    plist([3, 2, 1])
    """
    if not reverse:
        iterable = list(iterable)
        iterable.reverse()
    return reduce(lambda pl, elem: pl.cons(elem), iterable, _EMPTY_PLIST)