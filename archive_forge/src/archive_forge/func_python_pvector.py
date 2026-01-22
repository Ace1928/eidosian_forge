from abc import abstractmethod, ABCMeta
from collections.abc import Sequence, Hashable
from numbers import Integral
import operator
from typing import TypeVar, Generic
from pyrsistent._transformations import transform
def python_pvector(iterable=()):
    """
    Create a new persistent vector containing the elements in iterable.

    >>> v1 = pvector([1, 2, 3])
    >>> v1
    pvector([1, 2, 3])
    """
    return _EMPTY_PVECTOR.extend(iterable)