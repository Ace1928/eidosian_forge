import numpy as np
import pandas as pd
import patsy
import statsmodels.base.model as base
from statsmodels.regression.linear_model import OLS
import collections
from scipy.optimize import minimize
from statsmodels.iolib import summary2
from statsmodels.tools.numdiff import approx_fprime
import warnings
def covariance_group(self, group):
    ix = self.model._groups_ix[group]
    if len(ix) == 0:
        msg = "Group '%s' does not exist" % str(group)
        raise ValueError(msg)
    scale_data = self.model.exog_scale[ix, :]
    smooth_data = self.model.exog_smooth[ix, :]
    _, scale_names, smooth_names, _ = self.model._split_param_names()
    scale_data = pd.DataFrame(scale_data, columns=scale_names)
    smooth_data = pd.DataFrame(smooth_data, columns=smooth_names)
    time = self.model.time[ix]
    return self.model.covariance(time, self.scale_params, self.smooth_params, scale_data, smooth_data)