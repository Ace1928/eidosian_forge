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
def sresid(self):
    if self.scale == 0.0:
        sresid = self.resid.copy()
        sresid[:] = 0.0
        return sresid
    return self.resid / self.scale