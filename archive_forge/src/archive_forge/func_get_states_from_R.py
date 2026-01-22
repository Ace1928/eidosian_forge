from statsmodels.compat.pandas import QUARTER_END
from statsmodels.compat.platform import PLATFORM_LINUX32, PLATFORM_WIN
from itertools import product
import json
import pathlib
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import statsmodels.tsa.holtwinters as holtwinters
import statsmodels.tsa.statespace.exponential_smoothing as statespace
def get_states_from_R(results_R, k_states):
    if k_states > 1:
        xhat_R = results_R['states'][1:, 0:k_states]
    else:
        xhat_R = results_R['states'][1:]
        xhat_R = np.reshape(xhat_R, (len(xhat_R), 1))
    return xhat_R