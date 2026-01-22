import os
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
from statsmodels.tsa.stattools import bds
def test_pvalue(self):
    assert_almost_equal(self.res[1], self.pvalues, DECIMAL_8)