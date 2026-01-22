import warnings
from numbers import Integral, Real
import numpy as np
from scipy import optimize, sparse, stats
from scipy.special import boxcox
from ..base import (
from ..utils import _array_api, check_array
from ..utils._array_api import get_namespace
from ..utils._param_validation import Interval, Options, StrOptions, validate_params
from ..utils.extmath import _incremental_mean_and_var, row_norms
from ..utils.sparsefuncs import (
from ..utils.sparsefuncs_fast import (
from ..utils.validation import (
from ._encoders import OneHotEncoder
@validate_params({'X': ['array-like', 'sparse matrix'], 'axis': [Options(Integral, {0, 1})]}, prefer_skip_nested_validation=False)
def robust_scale(X, *, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True, unit_variance=False):
    """Standardize a dataset along any axis.

    Center to the median and component wise scale
    according to the interquartile range.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_sample, n_features)
        The data to center and scale.

    axis : int, default=0
        Axis used to compute the medians and IQR along. If 0,
        independently scale each feature, otherwise (if 1) scale
        each sample.

    with_centering : bool, default=True
        If `True`, center the data before scaling.

    with_scaling : bool, default=True
        If `True`, scale the data to unit variance (or equivalently,
        unit standard deviation).

    quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0,        default=(25.0, 75.0)
        Quantile range used to calculate `scale_`. By default this is equal to
        the IQR, i.e., `q_min` is the first quantile and `q_max` is the third
        quantile.

        .. versionadded:: 0.18

    copy : bool, default=True
        If False, try to avoid a copy and scale in place.
        This is not guaranteed to always work in place; e.g. if the data is
        a numpy array with an int dtype, a copy will be returned even with
        copy=False.

    unit_variance : bool, default=False
        If `True`, scale data so that normally distributed features have a
        variance of 1. In general, if the difference between the x-values of
        `q_max` and `q_min` for a standard normal distribution is greater
        than 1, the dataset will be scaled down. If less than 1, the dataset
        will be scaled up.

        .. versionadded:: 0.24

    Returns
    -------
    X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The transformed data.

    See Also
    --------
    RobustScaler : Performs centering and scaling using the Transformer API
        (e.g. as part of a preprocessing :class:`~sklearn.pipeline.Pipeline`).

    Notes
    -----
    This implementation will refuse to center scipy.sparse matrices
    since it would make them non-sparse and would potentially crash the
    program with memory exhaustion problems.

    Instead the caller is expected to either set explicitly
    `with_centering=False` (in that case, only variance scaling will be
    performed on the features of the CSR matrix) or to call `X.toarray()`
    if he/she expects the materialized dense array to fit in memory.

    To avoid memory copy the caller should pass a CSR matrix.

    For a comparison of the different scalers, transformers, and normalizers,
    see: :ref:`sphx_glr_auto_examples_preprocessing_plot_all_scaling.py`.

    .. warning:: Risk of data leak

        Do not use :func:`~sklearn.preprocessing.robust_scale` unless you know
        what you are doing. A common mistake is to apply it to the entire data
        *before* splitting into training and test sets. This will bias the
        model evaluation because information would have leaked from the test
        set to the training set.
        In general, we recommend using
        :class:`~sklearn.preprocessing.RobustScaler` within a
        :ref:`Pipeline <pipeline>` in order to prevent most risks of data
        leaking: `pipe = make_pipeline(RobustScaler(), LogisticRegression())`.

    Examples
    --------
    >>> from sklearn.preprocessing import robust_scale
    >>> X = [[-2, 1, 2], [-1, 0, 1]]
    >>> robust_scale(X, axis=0)  # scale each column independently
    array([[-1.,  1.,  1.],
           [ 1., -1., -1.]])
    >>> robust_scale(X, axis=1)  # scale each row independently
    array([[-1.5,  0. ,  0.5],
           [-1. ,  0. ,  1. ]])
    """
    X = check_array(X, accept_sparse=('csr', 'csc'), copy=False, ensure_2d=False, dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
    original_ndim = X.ndim
    if original_ndim == 1:
        X = X.reshape(X.shape[0], 1)
    s = RobustScaler(with_centering=with_centering, with_scaling=with_scaling, quantile_range=quantile_range, unit_variance=unit_variance, copy=copy)
    if axis == 0:
        X = s.fit_transform(X)
    else:
        X = s.fit_transform(X.T).T
    if original_ndim == 1:
        X = X.ravel()
    return X