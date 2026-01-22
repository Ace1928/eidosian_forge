from __future__ import annotations
import functools
from ._dtypes import _floating_dtypes, _numeric_dtypes
from ._manipulation_functions import reshape
from ._array_object import Array
from .._core.internal import _normalize_axis_indices as normalize_axis_tuple
from typing import TYPE_CHECKING
from typing import NamedTuple
import cupy as np
def svdvals(x: Array, /) -> Union[Array, Tuple[Array, ...]]:
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in svdvals')
    return Array._new(np.linalg.svd(x._array, compute_uv=False))