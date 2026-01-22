from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
def compute_all_knots(x, df, degree):
    order = degree + 1
    n_inner_knots = df - order
    lower_bound = np.min(x)
    upper_bound = np.max(x)
    knot_quantiles = np.linspace(0, 1, n_inner_knots + 2)[1:-1]
    inner_knots = _R_compat_quantile(x, knot_quantiles)
    all_knots = np.concatenate(([lower_bound, upper_bound] * order, inner_knots))
    return (all_knots, lower_bound, upper_bound, inner_knots)