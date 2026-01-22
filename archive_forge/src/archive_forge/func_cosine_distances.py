import itertools
import warnings
from functools import partial
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from scipy.sparse import csr_matrix, issparse
from scipy.spatial import distance
from .. import config_context
from ..exceptions import DataConversionWarning
from ..preprocessing import normalize
from ..utils import (
from ..utils._mask import _get_mask
from ..utils._param_validation import (
from ..utils.extmath import row_norms, safe_sparse_dot
from ..utils.fixes import parse_version, sp_base_version
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _num_samples, check_non_negative
from ._pairwise_distances_reduction import ArgKmin
from ._pairwise_fast import _chi2_kernel_fast, _sparse_manhattan
@validate_params({'X': ['array-like', 'sparse matrix'], 'Y': ['array-like', 'sparse matrix', None]}, prefer_skip_nested_validation=True)
def cosine_distances(X, Y=None):
    """Compute cosine distance between samples in X and Y.

    Cosine distance is defined as 1.0 minus the cosine similarity.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        Matrix `X`.

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features),             default=None
        Matrix `Y`.

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)
        Returns the cosine distance between samples in X and Y.

    See Also
    --------
    cosine_similarity : Compute cosine similarity between samples in X and Y.
    scipy.spatial.distance.cosine : Dense matrices only.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import cosine_distances
    >>> X = [[0, 0, 0], [1, 1, 1]]
    >>> Y = [[1, 0, 0], [1, 1, 0]]
    >>> cosine_distances(X, Y)
    array([[1.     , 1.     ],
           [0.42..., 0.18...]])
    """
    S = cosine_similarity(X, Y)
    S *= -1
    S += 1
    np.clip(S, 0, 2, out=S)
    if X is Y or Y is None:
        S[np.diag_indices_from(S)] = 0.0
    return S