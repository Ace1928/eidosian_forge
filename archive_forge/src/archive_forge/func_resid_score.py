import warnings
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lzip
from collections import defaultdict
import numpy as np
from statsmodels.graphics._regressionplots_doc import _plot_influence_doc
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import maybe_unwrap_results
def resid_score(self, joint=True, index=None, studentize=False):
    """Score observations scaled by inverse hessian.

        Score residual in resid_score are defined in analogy to a score test
        statistic for each observation.

        Parameters
        ----------
        joint : bool
            If joint is true, then a quadratic form similar to score_test is
            returned for each observation.
            If joint is false, then standardized score_obs are returned. The
            returned array is two-dimensional
        index : ndarray (optional)
            Optional index to select a subset of score_obs columns.
            By default, all columns of score_obs will be used.
        studentize : bool
            If studentize is true, the the scaled residuals are also
            studentized using the generalized leverage.

        Returns
        -------
        array :  1-D or 2-D residuals

        Notes
        -----
        Status: experimental

        Because of the one srep approacimation of d_params, score residuals
        are identical to cooks_distance, except for

        - cooks_distance is normalized by the number of parameters
        - cooks_distance uses cov_params, resid_score is based on Hessian.
          This will make them differ in the case of robust cov_params.

        """
    score_obs = self.results.model.score_obs(self.results.params)
    hess = self.results.model.hessian(self.results.params)
    if index is not None:
        score_obs = score_obs[:, index]
        hess = hess[index[:, None], index]
    if joint:
        resid = (score_obs.T * np.linalg.solve(-hess, score_obs.T)).sum(0)
    else:
        resid = score_obs / np.sqrt(np.diag(-hess))
    if studentize:
        if joint:
            resid /= np.sqrt(1 - self.hat_matrix_diag)
        else:
            resid /= np.sqrt(1 - self.hat_matrix_diag[:, None])
    return resid