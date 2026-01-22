from typing import Mapping, Optional, Sequence, Union, TYPE_CHECKING
import numpy
import numpy.typing as npt
import cupy
from cupy._core._scalar import get_typename
@classmethod
def from_ndarray(cls, x: cupy.ndarray) -> 'CArray':
    return CArray(x.dtype, x.ndim, x._c_contiguous, x._index_32_bits)