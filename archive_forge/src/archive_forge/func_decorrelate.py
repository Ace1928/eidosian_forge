import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from scipy.optimize import fminbound
import warnings
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (
def decorrelate(self, rhs):
    """
        Decorrelate the columns of `rhs`.

        Parameters
        ----------
        rhs : array_like
            A 2 dimensional array with the same number of rows as the
            PSD matrix represented by the class instance.

        Returns
        -------
        C^{-1/2} * rhs, where C is the covariance matrix represented
        by this class instance.

        Notes
        -----
        The returned matrix has the identity matrix as its row-wise
        population covariance matrix.

        This function exploits the factor structure for efficiency.
        """
    qval = -1 + 1 / np.sqrt(1 + self.scales)
    rhs = rhs / np.sqrt(self.diag)[:, None]
    rhs1 = np.dot(self.factor.T, rhs)
    rhs1 *= qval[:, None]
    rhs1 = np.dot(self.factor, rhs1)
    rhs += rhs1
    return rhs