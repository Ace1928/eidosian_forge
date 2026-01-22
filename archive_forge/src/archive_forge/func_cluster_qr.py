import warnings
from numbers import Integral, Real
import numpy as np
from scipy.linalg import LinAlgError, qr, svd
from scipy.sparse import csc_matrix
from ..base import BaseEstimator, ClusterMixin, _fit_context
from ..manifold import spectral_embedding
from ..metrics.pairwise import KERNEL_PARAMS, pairwise_kernels
from ..neighbors import NearestNeighbors, kneighbors_graph
from ..utils import as_float_array, check_random_state
from ..utils._param_validation import Interval, StrOptions, validate_params
from ._kmeans import k_means
def cluster_qr(vectors):
    """Find the discrete partition closest to the eigenvector embedding.

        This implementation was proposed in [1]_.

    .. versionadded:: 1.1

        Parameters
        ----------
        vectors : array-like, shape: (n_samples, n_clusters)
            The embedding space of the samples.

        Returns
        -------
        labels : array of integers, shape: n_samples
            The cluster labels of vectors.

        References
        ----------
        .. [1] :doi:`Simple, direct, and efficient multi-way spectral clustering, 2019
            Anil Damle, Victor Minden, Lexing Ying
            <10.1093/imaiai/iay008>`

    """
    k = vectors.shape[1]
    _, _, piv = qr(vectors.T, pivoting=True)
    ut, _, v = svd(vectors[piv[:k], :].T)
    vectors = abs(np.dot(vectors, np.dot(ut, v.conj())))
    return vectors.argmax(axis=1)