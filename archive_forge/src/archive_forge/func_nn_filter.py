import numpy as np
import scipy.sparse
from scipy.ndimage import median_filter
import sklearn.decomposition
from . import core
from ._cache import cache
from . import segment
from . import util
from .util.exceptions import ParameterError
from typing import Any, Callable, List, Optional, Tuple, Union
from ._typing import _IntLike_co, _FloatLike_co
@cache(level=30)
def nn_filter(S: np.ndarray, *, rec: Optional[Union[scipy.sparse.spmatrix, np.ndarray]]=None, aggregate: Optional[Callable]=None, axis: int=-1, **kwargs: Any) -> np.ndarray:
    """Filter by nearest-neighbor aggregation.

    Each data point (e.g, spectrogram column) is replaced
    by aggregating its nearest neighbors in feature space.

    This can be useful for de-noising a spectrogram or feature matrix.

    The non-local means method [#]_ can be recovered by providing a
    weighted recurrence matrix as input and specifying ``aggregate=np.average``.

    Similarly, setting ``aggregate=np.median`` produces sparse de-noising
    as in REPET-SIM [#]_.

    .. [#] Buades, A., Coll, B., & Morel, J. M.
        (2005, June). A non-local algorithm for image denoising.
        In Computer Vision and Pattern Recognition, 2005.
        CVPR 2005. IEEE Computer Society Conference on (Vol. 2, pp. 60-65). IEEE.

    .. [#] Rafii, Z., & Pardo, B.
        (2012, October).  "Music/Voice Separation Using the Similarity Matrix."
        International Society for Music Information Retrieval Conference, 2012.

    Parameters
    ----------
    S : np.ndarray
        The input data (spectrogram) to filter. Multi-channel is supported.

    rec : (optional) scipy.sparse.spmatrix or np.ndarray
        Optionally, a pre-computed nearest-neighbor matrix
        as provided by `librosa.segment.recurrence_matrix`

    aggregate : function
        aggregation function (default: `np.mean`)

        If ``aggregate=np.average``, then a weighted average is
        computed according to the (per-row) weights in ``rec``.

        For all other aggregation functions, all neighbors
        are treated equally.

    axis : int
        The axis along which to filter (by default, columns)

    **kwargs
        Additional keyword arguments provided to
        `librosa.segment.recurrence_matrix` if ``rec`` is not provided

    Returns
    -------
    S_filtered : np.ndarray
        The filtered data, with shape equivalent to the input ``S``.

    Raises
    ------
    ParameterError
        if ``rec`` is provided and its shape is incompatible with ``S``.

    See Also
    --------
    decompose
    hpss
    librosa.segment.recurrence_matrix

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    De-noise a chromagram by non-local median filtering.
    By default this would use euclidean distance to select neighbors,
    but this can be overridden directly by setting the ``metric`` parameter.

    >>> y, sr = librosa.load(librosa.ex('brahms'),
    ...                      offset=30, duration=10)
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    >>> chroma_med = librosa.decompose.nn_filter(chroma,
    ...                                          aggregate=np.median,
    ...                                          metric='cosine')

    To use non-local means, provide an affinity matrix and ``aggregate=np.average``.

    >>> rec = librosa.segment.recurrence_matrix(chroma, mode='affinity',
    ...                                         metric='cosine', sparse=True)
    >>> chroma_nlm = librosa.decompose.nn_filter(chroma, rec=rec,
    ...                                          aggregate=np.average)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=5, sharex=True, sharey=True, figsize=(10, 10))
    >>> librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='Unfiltered')
    >>> ax[0].label_outer()
    >>> librosa.display.specshow(chroma_med, y_axis='chroma', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Median-filtered')
    >>> ax[1].label_outer()
    >>> imgc = librosa.display.specshow(chroma_nlm, y_axis='chroma', x_axis='time', ax=ax[2])
    >>> ax[2].set(title='Non-local means')
    >>> ax[2].label_outer()
    >>> imgr1 = librosa.display.specshow(chroma - chroma_med,
    ...                          y_axis='chroma', x_axis='time', ax=ax[3])
    >>> ax[3].set(title='Original - median')
    >>> ax[3].label_outer()
    >>> imgr2 = librosa.display.specshow(chroma - chroma_nlm,
    ...                          y_axis='chroma', x_axis='time', ax=ax[4])
    >>> ax[4].label_outer()
    >>> ax[4].set(title='Original - NLM')
    >>> fig.colorbar(imgc, ax=ax[:3])
    >>> fig.colorbar(imgr1, ax=[ax[3]])
    >>> fig.colorbar(imgr2, ax=[ax[4]])
    """
    if aggregate is None:
        aggregate = np.mean
    rec_s: scipy.sparse.spmatrix
    if rec is None:
        kwargs = dict(kwargs)
        kwargs['sparse'] = True
        rec_s = segment.recurrence_matrix(S, axis=axis, **kwargs)
    elif not scipy.sparse.issparse(rec):
        rec_s = scipy.sparse.csc_matrix(rec)
    else:
        rec_s = rec
    if rec_s.shape[0] != S.shape[axis] or rec_s.shape[0] != rec_s.shape[1]:
        raise ParameterError(f'Invalid self-similarity matrix shape rec.shape={rec_s.shape} for S.shape={S.shape}')
    return __nn_filter_helper(rec_s.data, rec_s.indices, rec_s.indptr, S.swapaxes(0, axis), aggregate).swapaxes(0, axis)