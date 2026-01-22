import warnings
import numpy as np
import pandas as pd
import patsy
from scipy import sparse
from scipy.stats.distributions import norm
from statsmodels.base._penalties import Penalty
import statsmodels.base.model as base
from statsmodels.tools import data as data_tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
@cache_readonly
def random_effects(self):
    """
        The conditional means of random effects given the data.

        Returns
        -------
        random_effects : dict
            A dictionary mapping the distinct `group` values to the
            conditional means of the random effects for the group
            given the data.
        """
    try:
        cov_re_inv = np.linalg.inv(self.cov_re)
    except np.linalg.LinAlgError:
        raise ValueError('Cannot predict random effects from ' + 'singular covariance structure.')
    vcomp = self.vcomp
    k_re = self.k_re
    ranef_dict = {}
    for group_ix, group in enumerate(self.model.group_labels):
        endog = self.model.endog_li[group_ix]
        exog = self.model.exog_li[group_ix]
        ex_r = self.model._aex_r[group_ix]
        ex2_r = self.model._aex_r2[group_ix]
        vc_var = self.model._expand_vcomp(vcomp, group_ix)
        resid = endog
        if self.k_fe > 0:
            expval = np.dot(exog, self.fe_params)
            resid = resid - expval
        solver = _smw_solver(self.scale, ex_r, ex2_r, cov_re_inv, 1 / vc_var)
        vir = solver(resid)
        xtvir = _dot(ex_r.T, vir)
        xtvir[0:k_re] = np.dot(self.cov_re, xtvir[0:k_re])
        xtvir[k_re:] *= vc_var
        ranef_dict[group] = pd.Series(xtvir, index=self._expand_re_names(group_ix))
    return ranef_dict