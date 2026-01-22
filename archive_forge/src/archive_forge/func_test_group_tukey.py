from statsmodels.compat.python import asbytes
from io import BytesIO
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_, assert_allclose, assert_almost_equal, assert_equal, \
from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multicomp import (tukeyhsd, pairwise_tukeyhsd,
def test_group_tukey(self):
    res_t = get_thsd(self.mc, alpha=self.alpha)
    assert_almost_equal(res_t[4], self.confint2, decimal=2)