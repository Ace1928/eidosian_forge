import typing
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping, Sequence
from functools import lru_cache
from numbers import Number
from types import MappingProxyType
from typing import (
from pennylane.data.base import hdf5
from pennylane.data.base.hdf5 import HDF5, HDF5Any, HDF5Group
from pennylane.data.base.typing_util import UNSET, get_type, get_type_str
@classmethod
@lru_cache()
def bind_key(cls, __name: str) -> str:
    """Returns ``__name`` dot-prefixed with ``attrs_namespace``."""
    return '.'.join((cls.attrs_namespace, __name))