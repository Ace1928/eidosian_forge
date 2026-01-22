import numpy as np
from scipy.linalg import block_diag
from statsmodels.base._penalties import Penalty
def penalty_matrix(self, alpha=None):
    """penalty matrix for generalized additive model

        Parameters
        ----------
        alpha : list of floats or None
            penalty weights

        Returns
        -------
        penalty matrix
            block diagonal, square penalty matrix for quadratic penalization.
            The number of rows and columns are equal to the number of
            parameters in the regression model ``k_params``.

        Notes
        -----
        statsmodels does not support backwards compatibility when keywords are
        used as positional arguments. The order of keywords might change.
        We might need to add a ``params`` keyword if the need arises.
        """
    if alpha is None:
        alpha = self.alpha
    s_all = [np.zeros((self.start_idx, self.start_idx))]
    for i in range(self.k_variables):
        s_all.append(self.gp[i].penalty_matrix(alpha=alpha[i]))
    return block_diag(*s_all)