from enum import Enum
from abc import abstractmethod, ABCMeta
from collections.abc import Iterable
from typing import TypeVar, Generic
from pyrsistent._pmap import PMap, pmap
from pyrsistent._pset import PSet, pset
from pyrsistent._pvector import PythonPVector, python_pvector
def default_serializer(self, _, key, value):
    sk = key
    if isinstance(key, CheckedType):
        sk = key.serialize()
    sv = value
    if isinstance(value, CheckedType):
        sv = value.serialize()
    return (sk, sv)