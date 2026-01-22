import os
import numpy as np
from statsmodels.duration.survfunc import (
from numpy.testing import assert_allclose
import pandas as pd
import pytest
def test_survdiff_basic():
    ti = np.concatenate((ti1, ti2))
    st = np.concatenate((st1, st2))
    groups = np.ones(len(ti))
    groups[0:len(ti1)] = 0
    z, p = survdiff(ti, st, groups)
    assert_allclose(z, 2.14673, atol=0.0001, rtol=0.0001)
    assert_allclose(p, 0.14287, atol=0.0001, rtol=0.0001)