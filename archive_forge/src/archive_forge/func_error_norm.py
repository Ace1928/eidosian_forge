import warnings
import numpy as np
from scipy import linalg
from .. import config_context
from ..base import BaseEstimator, _fit_context
from ..metrics.pairwise import pairwise_distances
from ..utils import check_array
from ..utils._param_validation import validate_params
from ..utils.extmath import fast_logdet
def error_norm(self, comp_cov, norm='frobenius', scaling=True, squared=True):
    """Compute the Mean Squared Error between two covariance estimators.

        Parameters
        ----------
        comp_cov : array-like of shape (n_features, n_features)
            The covariance to compare with.

        norm : {"frobenius", "spectral"}, default="frobenius"
            The type of norm used to compute the error. Available error types:
            - 'frobenius' (default): sqrt(tr(A^t.A))
            - 'spectral': sqrt(max(eigenvalues(A^t.A))
            where A is the error ``(comp_cov - self.covariance_)``.

        scaling : bool, default=True
            If True (default), the squared error norm is divided by n_features.
            If False, the squared error norm is not rescaled.

        squared : bool, default=True
            Whether to compute the squared error norm or the error norm.
            If True (default), the squared error norm is returned.
            If False, the error norm is returned.

        Returns
        -------
        result : float
            The Mean Squared Error (in the sense of the Frobenius norm) between
            `self` and `comp_cov` covariance estimators.
        """
    error = comp_cov - self.covariance_
    if norm == 'frobenius':
        squared_norm = np.sum(error ** 2)
    elif norm == 'spectral':
        squared_norm = np.amax(linalg.svdvals(np.dot(error.T, error)))
    else:
        raise NotImplementedError('Only spectral and frobenius norms are implemented')
    if scaling:
        squared_norm = squared_norm / error.shape[0]
    if squared:
        result = squared_norm
    else:
        result = np.sqrt(squared_norm)
    return result