from abc import abstractmethod, ABCMeta
from collections.abc import Sequence, Hashable
from numbers import Integral
import operator
from typing import TypeVar, Generic
from pyrsistent._transformations import transform
def compare_pvector(v, other, operator):
    return operator(v.tolist(), other.tolist() if isinstance(other, PVector) else other)