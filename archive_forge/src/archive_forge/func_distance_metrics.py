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
def distance_metrics():
    """Valid metrics for pairwise_distances.

    This function simply returns the valid pairwise distance metrics.
    It exists to allow for a description of the mapping for
    each of the valid strings.

    The valid distance metrics, and the function they map to, are:

    =============== ========================================
    metric          Function
    =============== ========================================
    'cityblock'     metrics.pairwise.manhattan_distances
    'cosine'        metrics.pairwise.cosine_distances
    'euclidean'     metrics.pairwise.euclidean_distances
    'haversine'     metrics.pairwise.haversine_distances
    'l1'            metrics.pairwise.manhattan_distances
    'l2'            metrics.pairwise.euclidean_distances
    'manhattan'     metrics.pairwise.manhattan_distances
    'nan_euclidean' metrics.pairwise.nan_euclidean_distances
    =============== ========================================

    Read more in the :ref:`User Guide <metrics>`.

    Returns
    -------
    distance_metrics : dict
        Returns valid metrics for pairwise_distances.
    """
    return PAIRWISE_DISTANCE_FUNCTIONS