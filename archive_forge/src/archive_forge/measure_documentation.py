from . import select
from . import utils
from scipy import sparse
import numpy as np
import pandas as pd
import scipy.signal
Measure the number of cells in which each gene has non-negligible counts.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    cutoff : float, optional (default: 0)
        Number of counts above which expression is deemed non-negligible

    Returns
    -------
    capture-count : list-like, shape=[m_features]
        Capture count for each gene
    