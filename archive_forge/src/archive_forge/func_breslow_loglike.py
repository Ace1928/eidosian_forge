import numpy as np
from statsmodels.base import model
import statsmodels.base.model as base
from statsmodels.tools.decorators import cache_readonly
from statsmodels.compat.pandas import Appender
def breslow_loglike(self, params):
    """
        Returns the value of the log partial likelihood function
        evaluated at `params`, using the Breslow method to handle tied
        times.
        """
    surv = self.surv
    like = 0.0
    for stx in range(surv.nstrat):
        uft_ix = surv.ufailt_ix[stx]
        exog_s = surv.exog_s[stx]
        nuft = len(uft_ix)
        linpred = np.dot(exog_s, params)
        if surv.offset_s is not None:
            linpred += surv.offset_s[stx]
        linpred -= linpred.max()
        e_linpred = np.exp(linpred)
        xp0 = 0.0
        for i in range(nuft)[::-1]:
            ix = surv.risk_enter[stx][i]
            xp0 += e_linpred[ix].sum()
            ix = uft_ix[i]
            like += (linpred[ix] - np.log(xp0)).sum()
            ix = surv.risk_exit[stx][i]
            xp0 -= e_linpred[ix].sum()
    return like