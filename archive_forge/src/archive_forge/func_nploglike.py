import numpy as np
from statsmodels.base.model import Results
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
def nploglike(params, model):
    nobs = model.nobs
    pen_llf = alpha[k] * (1 - L1_wt) * np.sum(params ** 2) / 2
    llf = model.loglike(np.r_[params], **loglike_kwds)
    return -llf / nobs + pen_llf