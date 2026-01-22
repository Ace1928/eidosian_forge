from functools import lru_cache
from typing import Dict, Generic, Tuple, Type, TypeVar, Union, cast
import numpy as np
from scipy.sparse import (
from pennylane.data.base.attribute import AttributeInfo, DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Group
@property
def sparse_array_class(self) -> Type[SparseT]:
    """Returns the class of sparse array that will be returned by the ``get_value()``
        method."""
    return cast(Type[SparseT], self._supported_sparse_dict()[self.info['sparse_array_class']])