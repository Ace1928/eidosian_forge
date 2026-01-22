from enum import Enum
from abc import abstractmethod, ABCMeta
from collections.abc import Iterable
from typing import TypeVar, Generic
from pyrsistent._pmap import PMap, pmap
from pyrsistent._pset import PSet, pset
from pyrsistent._pvector import PythonPVector, python_pvector
def wrap_invariant(invariant):

    def f(*args, **kwargs):
        result = invariant(*args, **kwargs)
        if isinstance(result[0], bool):
            return result
        return _merge_invariant_results(result)
    return f