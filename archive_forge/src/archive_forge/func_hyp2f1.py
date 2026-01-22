import inspect
import warnings
import numpy as np
from scipy import special, stats
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.tools.sm_exceptions import (
from . import links as L, varfuncs as V
def hyp2f1(x):
    return special.hyp2f1(2 / 3.0, 1 / 3.0, 5 / 3.0, x)