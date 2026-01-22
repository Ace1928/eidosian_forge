from __future__ import annotations
import typing
import numpy as np
from .._utils import array_kind
def kde_statsmodels_m(data: FloatArray, grid: FloatArray, **kwargs: Any) -> FloatArray:
    """
    Multivariate Kernel Density Estimation with Statsmodels

    Parameters
    ----------
    data :
        Data points used to compute a density estimator. It
        has `n x p` dimensions, representing n points and p
        variables.
    grid :
        Data points at which the desity will be estimated. It
        has `m x p` dimensions, representing m points and p
        variables.

    Returns
    -------
    out :
        Density estimate. Has `m x 1` dimensions
    """
    from statsmodels.nonparametric.kernel_density import KDEMultivariate
    kde = KDEMultivariate(data, **kwargs)
    return kde.pdf(grid)