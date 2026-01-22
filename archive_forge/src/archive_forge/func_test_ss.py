from statsmodels.compat.python import lrange
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import t as student_t
from statsmodels.datasets import longley
from statsmodels.regression.linear_model import (
from statsmodels.tools.tools import add_constant
def test_ss(self):
    bse = np.array([r.bse for r in self.results])
    bse_1 = np.array([self.results[0].bse] * len(self.results))
    assert_allclose(bse, bse_1)