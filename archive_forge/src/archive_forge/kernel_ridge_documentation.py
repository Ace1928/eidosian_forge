from numbers import Real
import numpy as np
from .base import BaseEstimator, MultiOutputMixin, RegressorMixin, _fit_context
from .linear_model._ridge import _solve_cholesky_kernel
from .metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS, pairwise_kernels
from .utils._param_validation import Interval, StrOptions
from .utils.validation import _check_sample_weight, check_is_fitted
Predict using the kernel ridge model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples. If kernel == "precomputed" this is instead a
            precomputed kernel matrix, shape = [n_samples,
            n_samples_fitted], where n_samples_fitted is the number of
            samples used in the fitting for this estimator.

        Returns
        -------
        C : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        