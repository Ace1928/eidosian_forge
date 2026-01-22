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
def f_regression(X, y, *, center=True, force_finite=True):
    """Univariate linear regression tests returning F-statistic and p-values.

    Quick linear model for testing the effect of a single regressor,
    sequentially for many regressors.

    This is done in 2 steps:

    1. The cross correlation between each regressor and the target is computed
       using :func:`r_regression` as::

           E[(X[:, i] - mean(X[:, i])) * (y - mean(y))] / (std(X[:, i]) * std(y))

    2. It is converted to an F score and then to a p-value.

    :func:`f_regression` is derived from :func:`r_regression` and will rank
    features in the same order if all the features are positively correlated
    with the target.

    Note however that contrary to :func:`f_regression`, :func:`r_regression`
    values lie in [-1, 1] and can thus be negative. :func:`f_regression` is
    therefore recommended as a feature selection criterion to identify
    potentially predictive feature for a downstream classifier, irrespective of
    the sign of the association with the target variable.

    Furthermore :func:`f_regression` returns p-values while
    :func:`r_regression` does not.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

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
        Whether or not to force the F-statistics and associated p-values to
        be finite. There are two cases where the F-statistic is expected to not
        be finite:

        - when the target `y` or some features in `X` are constant. In this
          case, the Pearson's R correlation is not defined leading to obtain
          `np.nan` values in the F-statistic and p-value. When
          `force_finite=True`, the F-statistic is set to `0.0` and the
          associated p-value is set to `1.0`.
        - when a feature in `X` is perfectly correlated (or
          anti-correlated) with the target `y`. In this case, the F-statistic
          is expected to be `np.inf`. When `force_finite=True`, the F-statistic
          is set to `np.finfo(dtype).max` and the associated p-value is set to
          `0.0`.

        .. versionadded:: 1.1

    Returns
    -------
    f_statistic : ndarray of shape (n_features,)
        F-statistic for each feature.

    p_values : ndarray of shape (n_features,)
        P-values associated with the F-statistic.

    See Also
    --------
    r_regression: Pearson's R between label/feature for regression tasks.
    f_classif: ANOVA F-value between label/feature for classification tasks.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    SelectKBest: Select features based on the k highest scores.
    SelectFpr: Select features based on a false positive rate test.
    SelectFdr: Select features based on an estimated false discovery rate.
    SelectFwe: Select features based on family-wise error rate.
    SelectPercentile: Select features based on percentile of the highest
        scores.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.feature_selection import f_regression
    >>> X, y = make_regression(
    ...     n_samples=50, n_features=3, n_informative=1, noise=1e-4, random_state=42
    ... )
    >>> f_statistic, p_values = f_regression(X, y)
    >>> f_statistic
    array([1.2...+00, 2.6...+13, 2.6...+00])
    >>> p_values
    array([2.7..., 1.5..., 1.0...])
    """
    correlation_coefficient = r_regression(X, y, center=center, force_finite=force_finite)
    deg_of_freedom = y.size - (2 if center else 1)
    corr_coef_squared = correlation_coefficient ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        f_statistic = corr_coef_squared / (1 - corr_coef_squared) * deg_of_freedom
        p_values = stats.f.sf(f_statistic, 1, deg_of_freedom)
    if force_finite and (not np.isfinite(f_statistic).all()):
        mask_inf = np.isinf(f_statistic)
        f_statistic[mask_inf] = np.finfo(f_statistic.dtype).max
        mask_nan = np.isnan(f_statistic)
        f_statistic[mask_nan] = 0.0
        p_values[mask_nan] = 1.0
    return (f_statistic, p_values)