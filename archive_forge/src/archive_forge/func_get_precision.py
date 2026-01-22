import warnings
import numpy as np
from scipy import linalg
from .. import config_context
from ..base import BaseEstimator, _fit_context
from ..metrics.pairwise import pairwise_distances
from ..utils import check_array
from ..utils._param_validation import validate_params
from ..utils.extmath import fast_logdet
def get_precision(self):
    """Getter for the precision matrix.

        Returns
        -------
        precision_ : array-like of shape (n_features, n_features)
            The precision matrix associated to the current covariance object.
        """
    if self.store_precision:
        precision = self.precision_
    else:
        precision = linalg.pinvh(self.covariance_, check_finite=False)
    return precision