from statsmodels.compat.pandas import QUARTER_END, assert_index_equal
from statsmodels.compat.python import lrange
from io import BytesIO, StringIO
import os
import sys
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels.datasets import macrodata
import statsmodels.tools.data as data_util
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base.datetools import dates_from_str
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VAR, var_acf
def test_multiple_simulations(self):
    res0 = self.res0
    k_ar = res0.k_ar
    neqs = res0.neqs
    init = self.data[-k_ar:]
    sim1 = res0.simulate_var(seed=987128, steps=10)
    sim2 = res0.simulate_var(seed=987128, steps=10, nsimulations=2)
    assert_equal(sim2.shape, (2, 10, neqs))
    assert_allclose(sim1, sim2[0])
    sim2_init = res0.simulate_var(seed=987128, steps=10, initial_values=init, nsimulations=2)
    assert_allclose(sim2_init[0, :k_ar], init)
    assert_allclose(sim2_init[1, :k_ar], init)