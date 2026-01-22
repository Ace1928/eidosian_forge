import warnings
import numpy as np
import scipy.interpolate
import scipy.signal
from ..util.exceptions import ParameterError
from ..util import is_unique
from numpy.typing import ArrayLike
from typing import Callable, Optional, Sequence
def salience(S: np.ndarray, *, freqs: np.ndarray, harmonics: Sequence[float], weights: Optional[ArrayLike]=None, aggregate: Optional[Callable]=None, filter_peaks: bool=True, fill_value: float=np.nan, kind: str='linear', axis: int=-2) -> np.ndarray:
    """Harmonic salience function.

    Parameters
    ----------
    S : np.ndarray [shape=(..., d, n)]
        input time frequency magnitude representation (e.g. STFT or CQT magnitudes).
        Must be real-valued and non-negative.

    freqs : np.ndarray, shape=(S.shape[axis]) or shape=S.shape
        The frequency values corresponding to S's elements along the
        chosen axis.

        Frequencies can also be time-varying, e.g. as computed by
        `reassigned_spectrogram`, in which case the shape should
        match ``S``.

    harmonics : list-like, non-negative
        Harmonics to include in salience computation.  The first harmonic (1)
        corresponds to ``S`` itself. Values less than one (e.g., 1/2) correspond
        to sub-harmonics.

    weights : list-like
        The weight to apply to each harmonic in the summation. (default:
        uniform weights). Must be the same length as ``harmonics``.

    aggregate : function
        aggregation function (default: `np.average`)

        If ``aggregate=np.average``, then a weighted average is
        computed per-harmonic according to the specified weights.
        For all other aggregation functions, all harmonics
        are treated equally.

    filter_peaks : bool
        If true, returns harmonic summation only on frequencies of peak
        magnitude. Otherwise returns harmonic summation over the full spectrum.
        Defaults to True.

    fill_value : float
        The value to fill non-peaks in the output representation. (default:
        `np.nan`) Only used if ``filter_peaks == True``.

    kind : str
        Interpolation type for harmonic estimation.
        See `scipy.interpolate.interp1d`.

    axis : int
        The axis along which to compute harmonics

    Returns
    -------
    S_sal : np.ndarray
        ``S_sal`` will have the same shape as ``S``, and measure
        the overall harmonic energy at each frequency.

    See Also
    --------
    interp_harmonics

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'), duration=3)
    >>> S = np.abs(librosa.stft(y))
    >>> freqs = librosa.fft_frequencies(sr=sr)
    >>> harms = [1, 2, 3, 4]
    >>> weights = [1.0, 0.5, 0.33, 0.25]
    >>> S_sal = librosa.salience(S, freqs=freqs, harmonics=harms, weights=weights, fill_value=0)
    >>> print(S_sal.shape)
    (1025, 115)
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          sr=sr, y_axis='log', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='Magnitude spectrogram')
    >>> ax[0].label_outer()
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(S_sal,
    ...                                                        ref=np.max),
    ...                                sr=sr, y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Salience spectrogram')
    >>> fig.colorbar(img, ax=ax, format="%+2.0f dB")
    """
    if aggregate is None:
        aggregate = np.average
    if weights is None:
        weights = np.ones((len(harmonics),))
    else:
        weights = np.array(weights, dtype=float)
    S_harm = interp_harmonics(S, freqs=freqs, harmonics=harmonics, kind=kind, axis=axis)
    S_sal: np.ndarray
    if aggregate is np.average:
        S_sal = aggregate(S_harm, axis=axis - 1, weights=weights)
    else:
        S_sal = aggregate(S_harm, axis=axis - 1)
    if filter_peaks:
        S_peaks = scipy.signal.argrelmax(S, axis=axis)
        S_out = np.empty(S.shape)
        S_out.fill(fill_value)
        S_out[S_peaks] = S_sal[S_peaks]
        S_sal = S_out
    return S_sal