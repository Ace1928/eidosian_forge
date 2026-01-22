from functools import lru_cache
from typing import Dict, Generic, Tuple, Type, TypeVar, Union, cast
import numpy as np
from scipy.sparse import (
from pennylane.data.base.attribute import AttributeInfo, DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Group
@classmethod
def py_type(cls, value_type: Type[SparseArray]) -> str:
    """The module path of sparse array types is private, e.g ``scipy.sparse._csr.csr_array``.
        This method returns the public path e.g ``scipy.sparse.csr_array`` instead."""
    return f'scipy.sparse.{value_type.__qualname__}'