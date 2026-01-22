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
def test_hausman(self):
    res1, res2 = (self.res1, self.res2)
    hausm = res1.spec_hausman()
    assert_allclose(hausm[0], res2.hausman['DWH'], rtol=1e-11, atol=0)
    assert_allclose(hausm[1], res2.hausman['DWHp'], rtol=1e-10, atol=1e-25)