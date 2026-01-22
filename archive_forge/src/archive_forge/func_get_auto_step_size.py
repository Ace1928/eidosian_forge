import warnings
import numpy as np
from ..exceptions import ConvergenceWarning
from ..utils import check_array
from ..utils.extmath import row_norms
from ..utils.validation import _check_sample_weight
from ._base import make_dataset
from ._sag_fast import sag32, sag64
def get_auto_step_size(max_squared_sum, alpha_scaled, loss, fit_intercept, n_samples=None, is_saga=False):
    """Compute automatic step size for SAG solver.

    The step size is set to 1 / (alpha_scaled + L + fit_intercept) where L is
    the max sum of squares for over all samples.

    Parameters
    ----------
    max_squared_sum : float
        Maximum squared sum of X over samples.

    alpha_scaled : float
        Constant that multiplies the regularization term, scaled by
        1. / n_samples, the number of samples.

    loss : {'log', 'squared', 'multinomial'}
        The loss function used in SAG solver.

    fit_intercept : bool
        Specifies if a constant (a.k.a. bias or intercept) will be
        added to the decision function.

    n_samples : int, default=None
        Number of rows in X. Useful if is_saga=True.

    is_saga : bool, default=False
        Whether to return step size for the SAGA algorithm or the SAG
        algorithm.

    Returns
    -------
    step_size : float
        Step size used in SAG solver.

    References
    ----------
    Schmidt, M., Roux, N. L., & Bach, F. (2013).
    Minimizing finite sums with the stochastic average gradient
    https://hal.inria.fr/hal-00860051/document

    :arxiv:`Defazio, A., Bach F. & Lacoste-Julien S. (2014).
    "SAGA: A Fast Incremental Gradient Method With Support
    for Non-Strongly Convex Composite Objectives" <1407.0202>`
    """
    if loss in ('log', 'multinomial'):
        L = 0.25 * (max_squared_sum + int(fit_intercept)) + alpha_scaled
    elif loss == 'squared':
        L = max_squared_sum + int(fit_intercept) + alpha_scaled
    else:
        raise ValueError("Unknown loss function for SAG solver, got %s instead of 'log' or 'squared'" % loss)
    if is_saga:
        mun = min(2 * n_samples * alpha_scaled, L)
        step = 1.0 / (2 * L + mun)
    else:
        step = 1.0 / L
    return step