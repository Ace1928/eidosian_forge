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
def phasor(angles: Union[np.ndarray, _Real], *, mag: Optional[Union[np.ndarray, _Number]]=None) -> Union[np.ndarray, np.complex_]:
    """Construct a complex phasor representation from angles.

    When `mag` is not provided, this is equivalent to:

        z = np.cos(angles) + 1j * np.sin(angles)

    or by Euler's formula:

        z = np.exp(1j * angles)

    When `mag` is provided, this is equivalent to:

        z = mag * np.exp(1j * angles)

    This function should be more efficient (in time and memory) than the equivalent'
    formulations above, but produce numerically identical results.

    Parameters
    ----------
    angles : np.ndarray or scalar, real-valued
        Angle(s), measured in radians

    mag : np.ndarray or scalar, optional
        If provided, phasor(s) will be scaled by `mag`.

        If not provided (default), phasors will have unit magnitude.

        `mag` must be of compatible shape to multiply with `angles`.

    Returns
    -------
    z : np.ndarray or scalar, complex-valued
        Complex number(s) z corresponding to the given angle(s)
        and optional magnitude(s).

    Examples
    --------
    Construct unit phasors at angles 0, pi/2, and pi:

    >>> librosa.util.phasor([0, np.pi/2, np.pi])
    array([ 1.000e+00+0.000e+00j,  6.123e-17+1.000e+00j,
           -1.000e+00+1.225e-16j])

    Construct a phasor with magnitude 1/2:

    >>> librosa.util.phasor(np.pi/2, mag=0.5)
    (3.061616997868383e-17+0.5j)

    Or arrays of angles and magnitudes:

    >>> librosa.util.phasor(np.array([0, np.pi/2]), mag=np.array([0.5, 1.5]))
    array([5.000e-01+0.j , 9.185e-17+1.5j])
    """
    z = _phasor_angles(angles)
    if mag is not None:
        z *= mag
    return z