import numpy as np
from statsmodels.regression.linear_model import OLS, GLS, WLS
def atleast_2dcols(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    return x