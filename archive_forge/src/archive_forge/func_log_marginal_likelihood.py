import warnings
from numbers import Integral, Real
from operator import itemgetter
import numpy as np
import scipy.optimize
from scipy.linalg import cho_solve, cholesky, solve_triangular
from ..base import BaseEstimator, MultiOutputMixin, RegressorMixin, _fit_context, clone
from ..preprocessing._data import _handle_zeros_in_scale
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions
from ..utils.optimize import _check_optimize_result
from .kernels import RBF, Kernel
from .kernels import ConstantKernel as C
def log_marginal_likelihood(self, theta=None, eval_gradient=False, clone_kernel=True):
    """Return log-marginal likelihood of theta for training data.

        Parameters
        ----------
        theta : array-like of shape (n_kernel_params,) default=None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.

        eval_gradient : bool, default=False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.

        clone_kernel : bool, default=True
            If True, the kernel attribute is copied. If False, the kernel
            attribute is modified, but may result in a performance improvement.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.

        log_likelihood_gradient : ndarray of shape (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
    if theta is None:
        if eval_gradient:
            raise ValueError('Gradient can only be evaluated for theta!=None')
        return self.log_marginal_likelihood_value_
    if clone_kernel:
        kernel = self.kernel_.clone_with_theta(theta)
    else:
        kernel = self.kernel_
        kernel.theta = theta
    if eval_gradient:
        K, K_gradient = kernel(self.X_train_, eval_gradient=True)
    else:
        K = kernel(self.X_train_)
    K[np.diag_indices_from(K)] += self.alpha
    try:
        L = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
    except np.linalg.LinAlgError:
        return (-np.inf, np.zeros_like(theta)) if eval_gradient else -np.inf
    y_train = self.y_train_
    if y_train.ndim == 1:
        y_train = y_train[:, np.newaxis]
    alpha = cho_solve((L, GPR_CHOLESKY_LOWER), y_train, check_finite=False)
    log_likelihood_dims = -0.5 * np.einsum('ik,ik->k', y_train, alpha)
    log_likelihood_dims -= np.log(np.diag(L)).sum()
    log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
    log_likelihood = log_likelihood_dims.sum(axis=-1)
    if eval_gradient:
        inner_term = np.einsum('ik,jk->ijk', alpha, alpha)
        K_inv = cho_solve((L, GPR_CHOLESKY_LOWER), np.eye(K.shape[0]), check_finite=False)
        inner_term -= K_inv[..., np.newaxis]
        log_likelihood_gradient_dims = 0.5 * np.einsum('ijl,jik->kl', inner_term, K_gradient)
        log_likelihood_gradient = log_likelihood_gradient_dims.sum(axis=-1)
    if eval_gradient:
        return (log_likelihood, log_likelihood_gradient)
    else:
        return log_likelihood