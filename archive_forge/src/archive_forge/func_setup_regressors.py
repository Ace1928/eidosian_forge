from statsmodels.compat.pandas import FUTURE_STACK
from collections import defaultdict
import glob
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
def setup_regressors(df, low_pow=3, high_pow=3, cut=70, log=False):
    s = df.stack(**FUTURE_STACK).reset_index()
    q = s.level_0 / 10000
    y = stats.norm.ppf(q)
    cv = s[0]
    if log:
        cv = np.log(cv)
    m = np.where(s.level_0 <= df.index[cut])[0].max()
    reg = np.zeros((q.shape[0], 2 + low_pow + high_pow))
    reg[:m, 0] = 1
    for i in range(low_pow):
        reg[:m, i + 1] = cv[:m] ** (i + 1)
    w = 1 + low_pow
    reg[m:, w] = 1
    for i in range(high_pow):
        reg[m:, w + i + 1] = cv[m:] ** (i + 1)
    return (reg, y)