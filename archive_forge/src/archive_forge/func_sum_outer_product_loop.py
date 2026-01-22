import numpy as np
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.tools.grouputils import GroupSorted
def sum_outer_product_loop(x, group_iter):
    """sum outerproduct dot(x_i, x_i.T) over individuals

    loop version

    """
    mom = 0
    for g in group_iter():
        x_g = x[g]
        mom += np.outer(x_g, x_g)
    return mom