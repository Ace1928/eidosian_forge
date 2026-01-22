from statsmodels.compat.platform import PLATFORM_OSX
import os
import csv
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
import pytest
from statsmodels.regression.mixed_linear_model import (
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from statsmodels.base import _penalties as penalties
import statsmodels.tools.numdiff as nd
from .results import lme_r_results

    A class for holding various results obtained from fitting one data
    set using lmer in R.

    Parameters
    ----------
    meth : str
        Either "ml" or "reml".
    irfs : str
        Either "irf", for independent random effects, or "drf" for
        dependent random effects.
    ds_ix : int
        The number of the data set
    