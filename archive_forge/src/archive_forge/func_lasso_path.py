import numbers
import sys
import warnings
from abc import ABC, abstractmethod
from functools import partial
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from scipy import sparse
from ..base import MultiOutputMixin, RegressorMixin, _fit_context
from ..model_selection import check_cv
from ..utils import Bunch, check_array, check_scalar
from ..utils._metadata_requests import (
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import safe_sparse_dot
from ..utils.metadata_routing import (
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
from . import _cd_fast as cd_fast  # type: ignore
from ._base import LinearModel, _pre_fit, _preprocess_data
@validate_params({'X': ['array-like', 'sparse matrix'], 'y': ['array-like', 'sparse matrix'], 'eps': [Interval(Real, 0, None, closed='neither')], 'n_alphas': [Interval(Integral, 1, None, closed='left')], 'alphas': ['array-like', None], 'precompute': [StrOptions({'auto'}), 'boolean', 'array-like'], 'Xy': ['array-like', None], 'copy_X': ['boolean'], 'coef_init': ['array-like', None], 'verbose': ['verbose'], 'return_n_iter': ['boolean'], 'positive': ['boolean']}, prefer_skip_nested_validation=True)
def lasso_path(X, y, *, eps=0.001, n_alphas=100, alphas=None, precompute='auto', Xy=None, copy_X=True, coef_init=None, verbose=False, return_n_iter=False, positive=False, **params):
    """Compute Lasso path with coordinate descent.

    The Lasso optimization function varies for mono and multi-outputs.

    For mono-output tasks it is::

        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    For multi-output tasks it is::

        (1 / (2 * n_samples)) * ||Y - XW||^2_Fro + alpha * ||W||_21

    Where::

        ||W||_21 = \\sum_i \\sqrt{\\sum_j w_{ij}^2}

    i.e. the sum of norm of each row.

    Read more in the :ref:`User Guide <lasso>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication. If ``y`` is mono-output then ``X``
        can be sparse.

    y : {array-like, sparse matrix} of shape (n_samples,) or         (n_samples, n_targets)
        Target values.

    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, default=100
        Number of alphas along the regularization path.

    alphas : array-like, default=None
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically.

    precompute : 'auto', bool or array-like of shape             (n_features, n_features), default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    Xy : array-like of shape (n_features,) or (n_features, n_targets),         default=None
        Xy = np.dot(X.T, y) that can be precomputed. It is useful
        only when the Gram matrix is precomputed.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    coef_init : array-like of shape (n_features, ), default=None
        The initial values of the coefficients.

    verbose : bool or int, default=False
        Amount of verbosity.

    return_n_iter : bool, default=False
        Whether to return the number of iterations or not.

    positive : bool, default=False
        If set to True, forces coefficients to be positive.
        (Only allowed when ``y.ndim == 1``).

    **params : kwargs
        Keyword arguments passed to the coordinate descent solver.

    Returns
    -------
    alphas : ndarray of shape (n_alphas,)
        The alphas along the path where models are computed.

    coefs : ndarray of shape (n_features, n_alphas) or             (n_targets, n_features, n_alphas)
        Coefficients along the path.

    dual_gaps : ndarray of shape (n_alphas,)
        The dual gaps at the end of the optimization for each alpha.

    n_iters : list of int
        The number of iterations taken by the coordinate descent optimizer to
        reach the specified tolerance for each alpha.

    See Also
    --------
    lars_path : Compute Least Angle Regression or Lasso path using LARS
        algorithm.
    Lasso : The Lasso is a linear model that estimates sparse coefficients.
    LassoLars : Lasso model fit with Least Angle Regression a.k.a. Lars.
    LassoCV : Lasso linear model with iterative fitting along a regularization
        path.
    LassoLarsCV : Cross-validated Lasso using the LARS algorithm.
    sklearn.decomposition.sparse_encode : Estimator that can be used to
        transform signals into sparse linear combination of atoms from a fixed.

    Notes
    -----
    For an example, see
    :ref:`examples/linear_model/plot_lasso_coordinate_descent_path.py
    <sphx_glr_auto_examples_linear_model_plot_lasso_coordinate_descent_path.py>`.

    To avoid unnecessary memory duplication the X argument of the fit method
    should be directly passed as a Fortran-contiguous numpy array.

    Note that in certain cases, the Lars solver may be significantly
    faster to implement this functionality. In particular, linear
    interpolation can be used to retrieve model coefficients between the
    values output by lars_path

    Examples
    --------

    Comparing lasso_path and lars_path with interpolation:

    >>> import numpy as np
    >>> from sklearn.linear_model import lasso_path
    >>> X = np.array([[1, 2, 3.1], [2.3, 5.4, 4.3]]).T
    >>> y = np.array([1, 2, 3.1])
    >>> # Use lasso_path to compute a coefficient path
    >>> _, coef_path, _ = lasso_path(X, y, alphas=[5., 1., .5])
    >>> print(coef_path)
    [[0.         0.         0.46874778]
     [0.2159048  0.4425765  0.23689075]]

    >>> # Now use lars_path and 1D linear interpolation to compute the
    >>> # same path
    >>> from sklearn.linear_model import lars_path
    >>> alphas, active, coef_path_lars = lars_path(X, y, method='lasso')
    >>> from scipy import interpolate
    >>> coef_path_continuous = interpolate.interp1d(alphas[::-1],
    ...                                             coef_path_lars[:, ::-1])
    >>> print(coef_path_continuous([5., 1., .5]))
    [[0.         0.         0.46915237]
     [0.2159048  0.4425765  0.23668876]]
    """
    return enet_path(X, y, l1_ratio=1.0, eps=eps, n_alphas=n_alphas, alphas=alphas, precompute=precompute, Xy=Xy, copy_X=copy_X, coef_init=coef_init, verbose=verbose, positive=positive, return_n_iter=return_n_iter, **params)