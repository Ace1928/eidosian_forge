from statsmodels.compat.platform import PLATFORM_LINUX32
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from .results.results_discrete import RandHIE
from .test_discrete import CheckModelMixin
def compute_mixture_var(dispersion_factor, prob_main, mu):
    prob_infl = 1 - prob_main
    var = (dispersion_factor * (1 - prob_infl) * mu).mean()
    var += (((1 - prob_infl) * mu) ** 2).mean()
    var -= ((1 - prob_infl) * mu).mean() ** 2
    return var