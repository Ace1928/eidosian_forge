from typing import Any, Callable, Optional, Tuple
import warnings
import numpy as np
from scipy.stats import uniform, binom
def ecdf_confidence_band(ndraws: int, eval_points: np.ndarray, cdf_at_eval_points: np.ndarray, prob: float=0.95, method='simulated', **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the `prob`-level confidence band for the ECDF.

    Arguments
    ---------
    ndraws : int
        Number of samples in the original dataset.
    eval_points : np.ndarray
        Points at which the ECDF is evaluated. If these are dependent on the sample
        values, simultaneous confidence bands may not be correctly calibrated.
    cdf_at_eval_points : np.ndarray
        CDF values at the evaluation points.
    prob : float, default 0.95
        The target probability that a true ECDF lies within the confidence band.
    method : string, default "simulated"
        The method used to compute the confidence band. Valid options are:
        - "pointwise": Compute the pointwise (i.e. marginal) confidence band.
        - "simulated": Use Monte Carlo simulation to estimate a simultaneous confidence band.
          `rvs` must be provided.
    rvs: callable, optional
        A function that takes an integer `ndraws` and optionally the object passed to
        `random_state` and returns an array of `ndraws` samples from the same distribution
        as the original dataset. Required if `method` is "simulated" and variable is discrete.
    num_trials : int, default 1000
        The number of random ECDFs to generate for constructing simultaneous confidence bands
        (if `method` is "simulated").
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional
        If `None`, the `numpy.random.RandomState` singleton is used. If an `int`, a new
        ``numpy.random.RandomState`` instance is used, seeded with seed. If a `RandomState` or
        `Generator` instance, the instance is used.

    Returns
    -------
    prob_lower : np.ndarray
        Lower confidence band for the ECDF at the evaluation points.
    prob_upper : np.ndarray
        Upper confidence band for the ECDF at the evaluation points.
    """
    if not 0 < prob < 1:
        raise ValueError(f'Invalid value for `prob`. Expected 0 < prob < 1, but got {prob}.')
    if method == 'pointwise':
        prob_pointwise = prob
    elif method == 'simulated':
        prob_pointwise = _simulate_simultaneous_ecdf_band_probability(ndraws, eval_points, cdf_at_eval_points, prob=prob, **kwargs)
    else:
        raise ValueError(f"Unknown method {method}. Valid options are 'pointwise' or 'simulated'.")
    prob_lower, prob_upper = _get_pointwise_confidence_band(prob_pointwise, ndraws, cdf_at_eval_points)
    return (prob_lower, prob_upper)