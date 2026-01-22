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
def time_stretch(y: np.ndarray, *, rate: float, **kwargs: Any) -> np.ndarray:
    """Time-stretch an audio series by a fixed rate.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported.
    rate : float > 0 [scalar]
        Stretch factor.  If ``rate > 1``, then the signal is sped up.
        If ``rate < 1``, then the signal is slowed down.
    **kwargs : additional keyword arguments.
        See `librosa.decompose.stft` for details.

    Returns
    -------
    y_stretch : np.ndarray [shape=(..., round(n/rate))]
        audio time series stretched by the specified rate

    See Also
    --------
    pitch_shift :
        pitch shifting
    librosa.phase_vocoder :
        spectrogram phase vocoder
    pyrubberband.pyrb.time_stretch :
        high-quality time stretching using RubberBand

    Examples
    --------
    Compress to be twice as fast

    >>> y, sr = librosa.load(librosa.ex('choice'))
    >>> y_fast = librosa.effects.time_stretch(y, rate=2.0)

    Or half the original speed

    >>> y_slow = librosa.effects.time_stretch(y, rate=0.5)
    """
    if rate <= 0:
        raise ParameterError('rate must be a positive number')
    stft = core.stft(y, **kwargs)
    stft_stretch = core.phase_vocoder(stft, rate=rate, hop_length=kwargs.get('hop_length', None), n_fft=kwargs.get('n_fft', None))
    len_stretch = int(round(y.shape[-1] / rate))
    y_stretch = core.istft(stft_stretch, dtype=y.dtype, length=len_stretch, **kwargs)
    return y_stretch