import numbers
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Integral, Real
import numpy as np
from scipy import linalg, optimize, sparse
from scipy.sparse import linalg as sp_linalg
from ..base import MultiOutputMixin, RegressorMixin, _fit_context, is_classifier
from ..exceptions import ConvergenceWarning
from ..metrics import check_scoring, get_scorer_names
from ..model_selection import GridSearchCV
from ..preprocessing import LabelBinarizer
from ..utils import (
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import row_norms, safe_sparse_dot
from ..utils.fixes import _sparse_linalg_cg
from ..utils.metadata_routing import (
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._base import LinearClassifierMixin, LinearModel, _preprocess_data, _rescale_data
from ._sag import sag_solver
@validate_params({'X': ['array-like', 'sparse matrix', sp_linalg.LinearOperator], 'y': ['array-like'], 'alpha': [Interval(Real, 0, None, closed='left'), 'array-like'], 'sample_weight': [Interval(Real, None, None, closed='neither'), 'array-like', None], 'solver': [StrOptions({'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'})], 'max_iter': [Interval(Integral, 0, None, closed='left'), None], 'tol': [Interval(Real, 0, None, closed='left')], 'verbose': ['verbose'], 'positive': ['boolean'], 'random_state': ['random_state'], 'return_n_iter': ['boolean'], 'return_intercept': ['boolean'], 'check_input': ['boolean']}, prefer_skip_nested_validation=True)
def ridge_regression(X, y, alpha, *, sample_weight=None, solver='auto', max_iter=None, tol=0.0001, verbose=0, positive=False, random_state=None, return_n_iter=False, return_intercept=False, check_input=True):
    """Solve the ridge equation by the method of normal equations.

    Read more in the :ref:`User Guide <ridge_regression>`.

    Parameters
    ----------
    X : {array-like, sparse matrix, LinearOperator} of shape         (n_samples, n_features)
        Training data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.

    alpha : float or array-like of shape (n_targets,)
        Constant that multiplies the L2 term, controlling regularization
        strength. `alpha` must be a non-negative float i.e. in `[0, inf)`.

        When `alpha = 0`, the objective is equivalent to ordinary least
        squares, solved by the :class:`LinearRegression` object. For numerical
        reasons, using `alpha = 0` with the `Ridge` object is not advised.
        Instead, you should use the :class:`LinearRegression` object.

        If an array is passed, penalties are assumed to be specific to the
        targets. Hence they must correspond in number.

    sample_weight : float or array-like of shape (n_samples,), default=None
        Individual weights for each sample. If given a float, every sample
        will have the same weight. If sample_weight is not None and
        solver='auto', the solver will be set to 'cholesky'.

        .. versionadded:: 0.17

    solver : {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg',             'sag', 'saga', 'lbfgs'}, default='auto'
        Solver to use in the computational routines:

        - 'auto' chooses the solver automatically based on the type of data.

        - 'svd' uses a Singular Value Decomposition of X to compute the Ridge
          coefficients. It is the most stable solver, in particular more stable
          for singular matrices than 'cholesky' at the cost of being slower.

        - 'cholesky' uses the standard scipy.linalg.solve function to
          obtain a closed-form solution via a Cholesky decomposition of
          dot(X.T, X)

        - 'sparse_cg' uses the conjugate gradient solver as found in
          scipy.sparse.linalg.cg. As an iterative algorithm, this solver is
          more appropriate than 'cholesky' for large-scale data
          (possibility to set `tol` and `max_iter`).

        - 'lsqr' uses the dedicated regularized least-squares routine
          scipy.sparse.linalg.lsqr. It is the fastest and uses an iterative
          procedure.

        - 'sag' uses a Stochastic Average Gradient descent, and 'saga' uses
          its improved, unbiased version named SAGA. Both methods also use an
          iterative procedure, and are often faster than other solvers when
          both n_samples and n_features are large. Note that 'sag' and
          'saga' fast convergence is only guaranteed on features with
          approximately the same scale. You can preprocess the data with a
          scaler from sklearn.preprocessing.

        - 'lbfgs' uses L-BFGS-B algorithm implemented in
          `scipy.optimize.minimize`. It can be used only when `positive`
          is True.

        All solvers except 'svd' support both dense and sparse data. However, only
        'lsqr', 'sag', 'sparse_cg', and 'lbfgs' support sparse input when
        `fit_intercept` is True.

        .. versionadded:: 0.17
           Stochastic Average Gradient descent solver.
        .. versionadded:: 0.19
           SAGA solver.

    max_iter : int, default=None
        Maximum number of iterations for conjugate gradient solver.
        For the 'sparse_cg' and 'lsqr' solvers, the default value is determined
        by scipy.sparse.linalg. For 'sag' and saga solver, the default value is
        1000. For 'lbfgs' solver, the default value is 15000.

    tol : float, default=1e-4
        Precision of the solution. Note that `tol` has no effect for solvers 'svd' and
        'cholesky'.

        .. versionchanged:: 1.2
           Default value changed from 1e-3 to 1e-4 for consistency with other linear
           models.

    verbose : int, default=0
        Verbosity level. Setting verbose > 0 will display additional
        information depending on the solver used.

    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive.
        Only 'lbfgs' solver is supported in this case.

    random_state : int, RandomState instance, default=None
        Used when ``solver`` == 'sag' or 'saga' to shuffle the data.
        See :term:`Glossary <random_state>` for details.

    return_n_iter : bool, default=False
        If True, the method also returns `n_iter`, the actual number of
        iteration performed by the solver.

        .. versionadded:: 0.17

    return_intercept : bool, default=False
        If True and if X is sparse, the method also returns the intercept,
        and the solver is automatically changed to 'sag'. This is only a
        temporary fix for fitting the intercept with sparse data. For dense
        data, use sklearn.linear_model._preprocess_data before your regression.

        .. versionadded:: 0.17

    check_input : bool, default=True
        If False, the input arrays X and y will not be checked.

        .. versionadded:: 0.21

    Returns
    -------
    coef : ndarray of shape (n_features,) or (n_targets, n_features)
        Weight vector(s).

    n_iter : int, optional
        The actual number of iteration performed by the solver.
        Only returned if `return_n_iter` is True.

    intercept : float or ndarray of shape (n_targets,)
        The intercept of the model. Only returned if `return_intercept`
        is True and if X is a scipy sparse array.

    Notes
    -----
    This function won't compute the intercept.

    Regularization improves the conditioning of the problem and
    reduces the variance of the estimates. Larger values specify stronger
    regularization. Alpha corresponds to ``1 / (2C)`` in other linear
    models such as :class:`~sklearn.linear_model.LogisticRegression` or
    :class:`~sklearn.svm.LinearSVC`. If an array is passed, penalties are
    assumed to be specific to the targets. Hence they must correspond in
    number.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import ridge_regression
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(100, 4)
    >>> y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.1 * rng.standard_normal(100)
    >>> coef, intercept = ridge_regression(X, y, alpha=1.0, return_intercept=True)
    >>> list(coef)
    [1.9..., -1.0..., -0.0..., -0.0...]
    >>> intercept
    -0.0...
    """
    return _ridge_regression(X, y, alpha, sample_weight=sample_weight, solver=solver, max_iter=max_iter, tol=tol, verbose=verbose, positive=positive, random_state=random_state, return_n_iter=return_n_iter, return_intercept=return_intercept, X_scale=None, X_offset=None, check_input=check_input)