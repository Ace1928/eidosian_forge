import numpy as np
from statsmodels.base.model import Results
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
def npscore(params, model):
    nobs = model.nobs
    pen_grad = alpha[k] * (1 - L1_wt) * params
    gr = -model.score(np.r_[params], **score_kwds)[0] / nobs
    return gr + pen_grad