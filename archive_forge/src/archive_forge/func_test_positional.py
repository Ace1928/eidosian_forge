import copy
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose, assert_raises,
import pytest
import statsmodels.stats.power as smp
from statsmodels.stats.tests.test_weightstats import Holder
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
def test_positional(self):
    res1 = self.cls()
    args_names = ['effect_size', 'nobs', 'alpha', 'n_bins']
    kwds = copy.copy(self.kwds)
    del kwds['power']
    kwds.update(self.kwds_extra)
    args = [kwds[arg] for arg in args_names]
    if hasattr(self, 'decimal'):
        decimal = self.decimal
    else:
        decimal = 6
    assert_almost_equal(res1.power(*args), self.res2.power, decimal=decimal)