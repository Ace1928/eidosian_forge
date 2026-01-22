import warnings
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from statsmodels.base.data import PandasData
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, bic, hqic
from statsmodels.tools.numdiff import approx_fprime_cs, approx_hess_cs
from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tools.tools import Bunch, pinv_extended
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.regime_switching._hamilton_filter import (
from statsmodels.tsa.regime_switching._kim_smoother import (
from statsmodels.tsa.statespace.tools import (
def cy_hamilton_filter_log(initial_probabilities, regime_transition, conditional_loglikelihoods, model_order):
    """
    Hamilton filter in log space using Cython inner loop.

    Parameters
    ----------
    initial_probabilities : ndarray
        Array of initial probabilities, shaped (k_regimes,) giving the
        distribution of the regime process at time t = -order where order
        is a nonnegative integer.
    regime_transition : ndarray
        Matrix of regime transition probabilities, shaped either
        (k_regimes, k_regimes, 1) or if there are time-varying transition
        probabilities (k_regimes, k_regimes, nobs + order).  Entry [i, j,
        t] contains the probability of moving from j at time t-1 to i at
        time t, so each matrix regime_transition[:, :, t] should be left
        stochastic.  The first order entries and initial_probabilities are
        used to produce the initial joint distribution of dimension order +
        1 at time t=0.
    conditional_loglikelihoods : ndarray
        Array of loglikelihoods conditional on the last `order+1` regimes,
        shaped (k_regimes,)*(order + 1) + (nobs,).

    Returns
    -------
    filtered_marginal_probabilities : ndarray
        Array containing Pr[S_t=s_t | Y_t] - the probability of being in each
        regime conditional on time t information. Shaped (k_regimes, nobs).
    predicted_joint_probabilities : ndarray
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_{t-1}] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on time t-1
        information. Shaped (k_regimes,) * (order + 1) + (nobs,).
    joint_loglikelihoods : ndarray
        Array of loglikelihoods condition on time t information,
        shaped (nobs,).
    filtered_joint_probabilities : ndarray
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_{t}] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on time t
        information. Shaped (k_regimes,) * (order + 1) + (nobs,).
    """
    k_regimes = len(initial_probabilities)
    nobs = conditional_loglikelihoods.shape[-1]
    order = conditional_loglikelihoods.ndim - 2
    dtype = conditional_loglikelihoods.dtype
    incompatible_shapes = regime_transition.shape[-1] not in (1, nobs + model_order) or regime_transition.shape[:2] != (k_regimes, k_regimes) or conditional_loglikelihoods.shape[0] != k_regimes
    if incompatible_shapes:
        raise ValueError('Arguments do not have compatible shapes')
    initial_probabilities = np.log(initial_probabilities)
    regime_transition = np.log(np.maximum(regime_transition, 1e-20))
    filtered_marginal_probabilities = np.zeros((k_regimes, nobs), dtype=dtype)
    predicted_joint_probabilities = np.zeros((k_regimes,) * (order + 1) + (nobs,), dtype=dtype)
    joint_loglikelihoods = np.zeros((nobs,), dtype)
    filtered_joint_probabilities = np.zeros((k_regimes,) * (order + 1) + (nobs + 1,), dtype=dtype)
    filtered_marginal_probabilities[:, 0] = initial_probabilities
    tmp = np.copy(initial_probabilities)
    shape = (k_regimes, k_regimes)
    transition_t = 0
    for i in range(order):
        if regime_transition.shape[-1] > 1:
            transition_t = i
        tmp = np.reshape(regime_transition[..., transition_t], shape + (1,) * i) + tmp
    filtered_joint_probabilities[..., 0] = tmp
    if regime_transition.shape[-1] > 1:
        regime_transition = regime_transition[..., model_order:]
    prefix, dtype, _ = find_best_blas_type((regime_transition, conditional_loglikelihoods, joint_loglikelihoods, predicted_joint_probabilities, filtered_joint_probabilities))
    func = prefix_hamilton_filter_log_map[prefix]
    func(nobs, k_regimes, order, regime_transition, conditional_loglikelihoods.reshape(k_regimes ** (order + 1), nobs), joint_loglikelihoods, predicted_joint_probabilities.reshape(k_regimes ** (order + 1), nobs), filtered_joint_probabilities.reshape(k_regimes ** (order + 1), nobs + 1))
    predicted_joint_probabilities_log = predicted_joint_probabilities
    filtered_joint_probabilities_log = filtered_joint_probabilities
    predicted_joint_probabilities = np.exp(predicted_joint_probabilities)
    filtered_joint_probabilities = np.exp(filtered_joint_probabilities)
    filtered_marginal_probabilities = filtered_joint_probabilities[..., 1:]
    for i in range(1, filtered_marginal_probabilities.ndim - 1):
        filtered_marginal_probabilities = np.sum(filtered_marginal_probabilities, axis=-2)
    return (filtered_marginal_probabilities, predicted_joint_probabilities, joint_loglikelihoods, filtered_joint_probabilities[..., 1:], predicted_joint_probabilities_log, filtered_joint_probabilities_log[..., 1:])