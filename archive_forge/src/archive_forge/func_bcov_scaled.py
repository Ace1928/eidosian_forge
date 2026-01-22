import numpy as np
import scipy.stats as stats
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression._tools as reg_tools
import statsmodels.regression.linear_model as lm
import statsmodels.robust.norms as norms
import statsmodels.robust.scale as scale
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
@cache_readonly
def bcov_scaled(self):
    model = self.model
    m = np.mean(model.M.psi_deriv(self.sresid))
    var_psiprime = np.var(model.M.psi_deriv(self.sresid))
    k = 1 + (self.df_model + 1) / self.nobs * var_psiprime / m ** 2
    if model.cov == 'H1':
        ss_psi = np.sum(model.M.psi(self.sresid) ** 2)
        s_psi_deriv = np.sum(model.M.psi_deriv(self.sresid))
        return k ** 2 * (1 / self.df_resid * ss_psi * self.scale ** 2) / (1 / self.nobs * s_psi_deriv) ** 2 * model.normalized_cov_params
    else:
        W = np.dot(model.M.psi_deriv(self.sresid) * model.exog.T, model.exog)
        W_inv = np.linalg.inv(W)
        if model.cov == 'H2':
            return k * (1 / self.df_resid) * np.sum(model.M.psi(self.sresid) ** 2) * self.scale ** 2 / (1 / self.nobs * np.sum(model.M.psi_deriv(self.sresid))) * W_inv
        elif model.cov == 'H3':
            return k ** (-1) * 1 / self.df_resid * np.sum(model.M.psi(self.sresid) ** 2) * self.scale ** 2 * np.dot(np.dot(W_inv, np.dot(model.exog.T, model.exog)), W_inv)