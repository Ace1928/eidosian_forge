import numpy as np
import scipy.signal
from . import core
from . import decompose
from . import feature
from . import util
from .util.exceptions import ParameterError
from typing import Any, Callable, Iterable, Optional, Tuple, Union, overload
from typing_extensions import Literal
from numpy.typing import ArrayLike
def harmonic(y: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Extract harmonic elements from an audio time-series.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported.
    **kwargs : additional keyword arguments.
        See `librosa.decompose.hpss` for details.

    Returns
    -------
    y_harmonic : np.ndarray [shape=(..., n)]
        audio time series of just the harmonic portion

    See Also
    --------
    hpss : Separate harmonic and percussive components
    percussive : Extract only the percussive component
    librosa.decompose.hpss : HPSS for spectrograms

    Examples
    --------
    >>> # Extract harmonic component
    >>> y, sr = librosa.load(librosa.ex('choice'))
    >>> y_harmonic = librosa.effects.harmonic(y)

    >>> # Use a margin > 1.0 for greater harmonic separation
    >>> y_harmonic = librosa.effects.harmonic(y, margin=3.0)
    """
    stft = core.stft(y)
    stft_harm = decompose.hpss(stft, **kwargs)[0]
    y_harm = core.istft(stft_harm, dtype=y.dtype, length=y.shape[-1])
    return y_harm