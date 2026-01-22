import numpy as np
import pandas as pd
from ..doctools import document
from ..mapping.evaluation import after_stat
from .density import get_var_type, kde
from .stat import stat

    Compute density estimation for each point

    {usage}

    Parameters
    ----------
    {common_parameters}
    package : Literal["statsmodels", "scipy", "sklearn"], default="statsmodels"
        Package whose kernel density estimation to use.
    kde_params : dict, default=None
        Keyword arguments to pass on to the kde class.

    See Also
    --------
    statsmodels.nonparametric.kde.KDEMultivariate
    scipy.stats.gaussian_kde
    sklearn.neighbors.KernelDensity
    