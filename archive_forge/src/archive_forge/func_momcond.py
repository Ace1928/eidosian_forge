from statsmodels.compat.python import lrange, lmap
import os
import copy
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
import statsmodels.sandbox.regression.gmm as gmm
def momcond(self, params):
    p0, p1, p2, p3 = params
    endog = self.endog[:, None]
    exog = self.exog
    inst = self.instrument
    mom0 = (endog - p0 - p1 * exog) * inst
    mom1 = ((endog - p0 - p1 * exog) ** 2 - p2 * exog ** (2 * p3) / 12) * inst
    g = np.column_stack((mom0, mom1))
    return g