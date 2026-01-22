import os
import numpy as np
from statsmodels.duration.survfunc import (
from numpy.testing import assert_allclose
import pandas as pd
import pytest
def test_survdiff_entry_2():
    ti = np.r_[5, 3, 4, 2, 5, 4, 6, 7, 5, 9]
    st = np.r_[1, 1, 0, 1, 1, 0, 1, 1, 0, 0]
    gr = np.r_[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    entry = np.r_[1, 2, 2, 1, 3, 3, 5, 4, 2, 5]
    z, p = survdiff(ti, st, gr)
    assert_allclose(z, 6.694424)
    assert_allclose(p, 0.00967149)
    z, p = survdiff(ti, st, gr, entry=entry)
    assert_allclose(z, 3.0)
    assert_allclose(p, 0.083264516)