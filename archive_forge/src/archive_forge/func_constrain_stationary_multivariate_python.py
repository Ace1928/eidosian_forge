import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def constrain_stationary_multivariate_python(unconstrained, error_variance, transform_variance=False, prefix=None):
    """
    Transform unconstrained parameters used by the optimizer to constrained
    parameters used in likelihood evaluation for a vector autoregression.

    Parameters
    ----------
    unconstrained : array or list
        Arbitrary matrices to be transformed to stationary coefficient matrices
        of the VAR. If a list, should be a list of length `order`, where each
        element is an array sized `k_endog` x `k_endog`. If an array, should be
        the matrices horizontally concatenated and sized
        `k_endog` x `k_endog * order`.
    error_variance : ndarray
        The variance / covariance matrix of the error term. Should be sized
        `k_endog` x `k_endog`. This is used as input in the algorithm even if
        is not transformed by it (when `transform_variance` is False). The
        error term variance is required input when transformation is used
        either to force an autoregressive component to be stationary or to
        force a moving average component to be invertible.
    transform_variance : bool, optional
        Whether or not to transform the error variance term. This option is
        not typically used, and the default is False.
    prefix : {'s','d','c','z'}, optional
        The appropriate BLAS prefix to use for the passed datatypes. Only
        use if absolutely sure that the prefix is correct or an error will
        result.

    Returns
    -------
    constrained : array or list
        Transformed coefficient matrices leading to a stationary VAR
        representation. Will match the type of the passed `unconstrained`
        variable (so if a list was passed, a list will be returned).

    Notes
    -----
    In the notation of [1]_, the arguments `(variance, unconstrained)` are
    written as :math:`(\\Sigma, A_1, \\dots, A_p)`, where :math:`p` is the order
    of the vector autoregression, and is here determined by the length of
    the `unconstrained` argument.

    There are two steps in the constraining algorithm.

    First, :math:`(A_1, \\dots, A_p)` are transformed into
    :math:`(P_1, \\dots, P_p)` via Lemma 2.2 of [1]_.

    Second, :math:`(\\Sigma, P_1, \\dots, P_p)` are transformed into
    :math:`(\\Sigma, \\phi_1, \\dots, \\phi_p)` via Lemmas 2.1 and 2.3 of [1]_.

    If `transform_variance=True`, then only Lemma 2.1 is applied in the second
    step.

    While this function can be used even in the univariate case, it is much
    slower, so in that case `constrain_stationary_univariate` is preferred.

    References
    ----------
    .. [1] Ansley, Craig F., and Robert Kohn. 1986.
       "A Note on Reparameterizing a Vector Autoregressive Moving Average Model
       to Enforce Stationarity."
       Journal of Statistical Computation and Simulation 24 (2): 99-106.
    .. [*] Ansley, Craig F, and Paul Newbold. 1979.
       "Multivariate Partial Autocorrelations."
       In Proceedings of the Business and Economic Statistics Section, 349-53.
       American Statistical Association
    """
    use_list = type(unconstrained) is list
    if not use_list:
        k_endog, order = unconstrained.shape
        order //= k_endog
        unconstrained = [unconstrained[:k_endog, i * k_endog:(i + 1) * k_endog] for i in range(order)]
    order = len(unconstrained)
    k_endog = unconstrained[0].shape[0]
    sv_constrained = _constrain_sv_less_than_one_python(unconstrained, order, k_endog)
    constrained, var = _compute_coefficients_from_multivariate_pacf_python(sv_constrained, error_variance, transform_variance, order, k_endog)
    if not use_list:
        constrained = np.concatenate(constrained, axis=1).reshape(k_endog, k_endog * order)
    return (constrained, var)