import warnings
from numbers import Integral, Real
import numpy as np
from scipy import special, stats
from scipy.sparse import issparse
from ..base import BaseEstimator, _fit_context
from ..preprocessing import LabelBinarizer
from ..utils import as_float_array, check_array, check_X_y, safe_mask, safe_sqr
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import row_norms, safe_sparse_dot
from ..utils.validation import check_is_fitted
from ._base import SelectorMixin
@validate_params({'X': ['array-like', 'sparse matrix'], 'y': ['array-like'], 'center': ['boolean'], 'force_finite': ['boolean']}, prefer_skip_nested_validation=True)
def r_regression(X, y, *, center=True, force_finite=True):
    """Compute Pearson's r for each features and the target.

    Pearson's r is also known as the Pearson correlation coefficient.

    Linear model for testing the individual effect of each of many regressors.
    This is a scoring function to be used in a feature selection procedure, not
    a free standing feature selection procedure.

    The cross correlation between each regressor and the target is computed
    as::

        E[(X[:, i] - mean(X[:, i])) * (y - mean(y))] / (std(X[:, i]) * std(y))

    For more on usage see the :ref:`User Guide <univariate_feature_selection>`.

    .. versionadded:: 1.0

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data matrix.

    y : array-like of shape (n_samples,)
        The target vector.

    center : bool, default=True
        Whether or not to center the data matrix `X` and the target vector `y`.
        By default, `X` and `y` will be centered.

    force_finite : bool, default=True
        Whether or not to force the Pearson's R correlation to be finite.
        In the particular case where some features in `X` or the target `y`
        are constant, the Pearson's R correlation is not defined. When
        `force_finite=False`, a correlation of `np.nan` is returned to
        acknowledge this case. When `force_finite=True`, this value will be
        forced to a minimal correlation of `0.0`.

        .. versionadded:: 1.1

    Returns
    -------
    correlation_coefficient : ndarray of shape (n_features,)
        Pearson's R correlation coefficients of features.

    See Also
    --------
    f_regression: Univariate linear regression tests returning f-statistic
        and p-values.
    mutual_info_regression: Mutual information for a continuous target.
    f_classif: ANOVA F-value between label/feature for classification tasks.
    chi2: Chi-squared stats of non-negative features for classification tasks.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.feature_selection import r_regression
    >>> X, y = make_regression(
    ...     n_samples=50, n_features=3, n_informative=1, noise=1e-4, random_state=42
    ... )
    >>> r_regression(X, y)
    array([-0.15...,  1.        , -0.22...])
    """
    X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'], dtype=np.float64)
    n_samples = X.shape[0]
    if center:
        y = y - np.mean(y)
        X_means = X.mean(axis=0)
        X_means = X_means.getA1() if isinstance(X_means, np.matrix) else X_means
        X_norms = np.sqrt(row_norms(X.T, squared=True) - n_samples * X_means ** 2)
    else:
        X_norms = row_norms(X.T)
    correlation_coefficient = safe_sparse_dot(y, X)
    with np.errstate(divide='ignore', invalid='ignore'):
        correlation_coefficient /= X_norms
        correlation_coefficient /= np.linalg.norm(y)
    if force_finite and (not np.isfinite(correlation_coefficient).all()):
        nan_mask = np.isnan(correlation_coefficient)
        correlation_coefficient[nan_mask] = 0.0
    return correlation_coefficient