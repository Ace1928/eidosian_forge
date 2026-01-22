from statsmodels.compat.pandas import Appender
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as spl
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import bool_like
def fitfunc(a):
    dif = residmat[:, 0] - a ** designx * residmat[:, 1]
    return np.dot(dif ** 2, wts)