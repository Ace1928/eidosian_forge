from statsmodels.compat.pandas import FUTURE_STACK
import numpy as np
import pandas as pd
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
def get_sample(model):
    if model._index_dates:
        mask = ~np.isnan(model.endog).all(axis=1)
        ix = model._index[mask]
        d = ix[0]
        sample = ['%s' % d]
        d = ix[-1]
        sample += ['- ' + '%s' % d]
    else:
        sample = [str(0), ' - ' + str(model.nobs)]
    return sample