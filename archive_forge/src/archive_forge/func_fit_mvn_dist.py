import itertools
from pyomo.common.dependencies import (
from pyomo.common.dependencies.matplotlib import pyplot as plt
from pyomo.common.dependencies.scipy import stats
imports_available = (
def fit_mvn_dist(theta_values):
    """
    Fit a multivariate normal distribution to theta values

    Parameters
    ----------
    theta_values: DataFrame
        Theta values, columns = variable names

    Returns
    ---------
    scipy.stats.multivariate_normal distribution
    """
    assert isinstance(theta_values, pd.DataFrame)
    dist = stats.multivariate_normal(theta_values.mean(), theta_values.cov(), allow_singular=True)
    return dist