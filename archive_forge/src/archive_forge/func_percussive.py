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
def percussive(y: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Extract percussive elements from an audio time-series.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported.
    **kwargs : additional keyword arguments.
        See `librosa.decompose.hpss` for details.

    Returns
    -------
    y_percussive : np.ndarray [shape=(..., n)]
        audio time series of just the percussive portion

    See Also
    --------
    hpss : Separate harmonic and percussive components
    harmonic : Extract only the harmonic component
    librosa.decompose.hpss : HPSS for spectrograms

    Examples
    --------
    >>> # Extract percussive component
    >>> y, sr = librosa.load(librosa.ex('choice'))
    >>> y_percussive = librosa.effects.percussive(y)

    >>> # Use a margin > 1.0 for greater percussive separation
    >>> y_percussive = librosa.effects.percussive(y, margin=3.0)
    """
    stft = core.stft(y)
    stft_perc = decompose.hpss(stft, **kwargs)[1]
    y_perc = core.istft(stft_perc, dtype=y.dtype, length=y.shape[-1])
    return y_perc