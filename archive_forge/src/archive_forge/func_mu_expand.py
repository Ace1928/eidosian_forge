from __future__ import annotations
import os
import pathlib
import warnings
import soundfile as sf
import audioread
import numpy as np
import scipy.signal
import soxr
import lazy_loader as lazy
from numba import jit, stencil, guvectorize
from .fft import get_fftlib
from .convert import frames_to_samples, time_to_samples
from .._cache import cache
from .. import util
from ..util.exceptions import ParameterError
from ..util.decorators import deprecated
from ..util.deprecation import Deprecated, rename_kw
from .._typing import _FloatLike_co, _IntLike_co, _SequenceLike
from typing import Any, BinaryIO, Callable, Generator, Optional, Tuple, Union, List
from numpy.typing import DTypeLike, ArrayLike
def mu_expand(x: Union[np.ndarray, _FloatLike_co], *, mu: float=255.0, quantize: bool=True) -> np.ndarray:
    """mu-law expansion

    This function is the inverse of ``mu_compress``. Given a mu-law compressed
    signal ``-1 <= x <= 1``, the mu-law expansion is calculated by::

        sign(x) * (1 / mu) * ((1 + mu)**abs(x) - 1)

    Parameters
    ----------
    x : np.ndarray
        The compressed signal.
        If ``quantize=True``, values must be in the range [-1, +1].
    mu : positive number
        The compression parameter.  Values of the form ``2**n - 1``
        (e.g., 15, 31, 63, etc.) are most common.
    quantize : boolean
        If ``True``, the input is assumed to be quantized to
        ``1 + mu`` distinct integer values.

    Returns
    -------
    x_expanded : np.ndarray with values in the range [-1, +1]
        The mu-law expanded signal.

    Raises
    ------
    ParameterError
        If ``x`` has values outside the range [-1, +1] and ``quantize=False``
        If ``mu <= 0``

    See Also
    --------
    mu_compress

    Examples
    --------
    Compress and expand without quantization

    >>> x = np.linspace(-1, 1, num=16)
    >>> x
    array([-1.        , -0.86666667, -0.73333333, -0.6       , -0.46666667,
           -0.33333333, -0.2       , -0.06666667,  0.06666667,  0.2       ,
            0.33333333,  0.46666667,  0.6       ,  0.73333333,  0.86666667,
            1.        ])
    >>> y = librosa.mu_compress(x, quantize=False)
    >>> y
    array([-1.        , -0.97430198, -0.94432361, -0.90834832, -0.86336132,
           -0.80328309, -0.71255496, -0.52124063,  0.52124063,  0.71255496,
            0.80328309,  0.86336132,  0.90834832,  0.94432361,  0.97430198,
            1.        ])
    >>> z = librosa.mu_expand(y, quantize=False)
    >>> z
    array([-1.        , -0.86666667, -0.73333333, -0.6       , -0.46666667,
           -0.33333333, -0.2       , -0.06666667,  0.06666667,  0.2       ,
            0.33333333,  0.46666667,  0.6       ,  0.73333333,  0.86666667,
            1.        ])

    Compress and expand with quantization.  Note that this necessarily
    incurs quantization error, particularly for values near +-1.

    >>> y = librosa.mu_compress(x, quantize=True)
    >>> y
    array([-128, -124, -120, -116, -110, -102,  -91,  -66,   66,   91,  102,
            110,  116,  120,  124,  127])
    >>> z = librosa.mu_expand(y, quantize=True)
    array([-1.        , -0.84027248, -0.70595818, -0.59301377, -0.4563785 ,
           -0.32155973, -0.19817918, -0.06450245,  0.06450245,  0.19817918,
            0.32155973,  0.4563785 ,  0.59301377,  0.70595818,  0.84027248,
            0.95743702])
    """
    if mu <= 0:
        raise ParameterError(f'Inverse mu-law compression parameter mu={mu} must be strictly positive.')
    if quantize:
        x = x * 2.0 / (1 + mu)
    if np.any(x < -1) or np.any(x > 1):
        raise ParameterError(f'Inverse mu-law input x={x} must be in the range [-1, +1].')
    return np.sign(x) / mu * (np.power(1 + mu, np.abs(x)) - 1)