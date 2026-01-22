import numbers
import numpy as np
from scipy.special import xlogy
from ..utils import check_scalar
from ..utils.stats import _weighted_percentile
from ._loss import (
from .link import (
def init_gradient_and_hessian(self, n_samples, dtype=np.float64, order='F'):
    """Initialize arrays for gradients and hessians.

        Unless hessians are constant, arrays are initialized with undefined values.

        Parameters
        ----------
        n_samples : int
            The number of samples, usually passed to `fit()`.
        dtype : {np.float64, np.float32}, default=np.float64
            The dtype of the arrays gradient and hessian.
        order : {'C', 'F'}, default='F'
            Order of the arrays gradient and hessian. The default 'F' makes the arrays
            contiguous along samples.

        Returns
        -------
        gradient : C-contiguous array of shape (n_samples,) or array of shape             (n_samples, n_classes)
            Empty array (allocated but not initialized) to be used as argument
            gradient_out.
        hessian : C-contiguous array of shape (n_samples,), array of shape
            (n_samples, n_classes) or shape (1,)
            Empty (allocated but not initialized) array to be used as argument
            hessian_out.
            If constant_hessian is True (e.g. `HalfSquaredError`), the array is
            initialized to ``1``.
        """
    if dtype not in (np.float32, np.float64):
        raise ValueError(f"Valid options for 'dtype' are np.float32 and np.float64. Got dtype={dtype} instead.")
    if self.is_multiclass:
        shape = (n_samples, self.n_classes)
    else:
        shape = (n_samples,)
    gradient = np.empty(shape=shape, dtype=dtype, order=order)
    if self.constant_hessian:
        hessian = np.ones(shape=(1,), dtype=dtype)
    else:
        hessian = np.empty(shape=shape, dtype=dtype, order=order)
    return (gradient, hessian)