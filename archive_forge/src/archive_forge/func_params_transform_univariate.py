import numpy as np
from scipy import stats
import pandas as pd
def params_transform_univariate(params, cov_params, link=None, transform=None, row_labels=None):
    """
    results for univariate, nonlinear, monotonicaly transformed parameters

    This provides transformed values, standard errors and confidence interval
    for transformations of parameters, for example in calculating rates with
    `exp(params)` in the case of Poisson or other models with exponential
    mean function.
    """
    from statsmodels.genmod.families import links
    if link is None and transform is None:
        link = links.Log()
    if row_labels is None and hasattr(params, 'index'):
        row_labels = params.index
    params = np.asarray(params)
    predicted_mean = link.inverse(params)
    link_deriv = link.inverse_deriv(params)
    var_pred_mean = link_deriv ** 2 * np.diag(cov_params)
    dist = stats.norm
    linpred = PredictionResultsMean(params, np.diag(cov_params), dist=dist, row_labels=row_labels, link=links.Identity())
    res = PredictionResultsMean(predicted_mean, var_pred_mean, dist=dist, row_labels=row_labels, linpred=linpred, link=link)
    return res