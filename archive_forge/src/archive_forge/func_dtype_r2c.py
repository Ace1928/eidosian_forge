from __future__ import annotations
import scipy.ndimage
import scipy.sparse
import numpy as np
import numba
from numpy.lib.stride_tricks import as_strided
from .._cache import cache
from .exceptions import ParameterError
from .deprecation import Deprecated
from numpy.typing import ArrayLike, DTypeLike
from typing import (
from typing_extensions import Literal
from .._typing import _SequenceLike, _FloatLike_co, _ComplexLike_co
def dtype_r2c(d: DTypeLike, *, default: Optional[type]=np.complex64) -> DTypeLike:
    """Find the complex numpy dtype corresponding to a real dtype.

    This is used to maintain numerical precision and memory footprint
    when constructing complex arrays from real-valued data
    (e.g. in a Fourier transform).

    A `float32` (single-precision) type maps to `complex64`,
    while a `float64` (double-precision) maps to `complex128`.

    Parameters
    ----------
    d : np.dtype
        The real-valued dtype to convert to complex.
        If ``d`` is a complex type already, it will be returned.
    default : np.dtype, optional
        The default complex target type, if ``d`` does not match a
        known dtype

    Returns
    -------
    d_c : np.dtype
        The complex dtype

    See Also
    --------
    dtype_c2r
    numpy.dtype

    Examples
    --------
    >>> librosa.util.dtype_r2c(np.float32)
    dtype('complex64')

    >>> librosa.util.dtype_r2c(np.int16)
    dtype('complex64')

    >>> librosa.util.dtype_r2c(np.complex128)
    dtype('complex128')
    """
    mapping: Dict[DTypeLike, type] = {np.dtype(np.float32): np.complex64, np.dtype(np.float64): np.complex128, np.dtype(float): np.dtype(complex).type}
    dt = np.dtype(d)
    if dt.kind == 'c':
        return dt
    return np.dtype(mapping.get(dt, default))