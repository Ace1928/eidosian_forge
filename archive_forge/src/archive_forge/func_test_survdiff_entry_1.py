import os
import numpy as np
from statsmodels.duration.survfunc import (
from numpy.testing import assert_allclose
import pandas as pd
import pytest
def test_survdiff_entry_1():
    ti = np.r_[1, 3, 4, 2, 5, 4, 6, 7, 5, 9]
    st = np.r_[1, 1, 0, 1, 1, 0, 1, 1, 0, 0]
    gr = np.r_[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    entry = np.r_[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    z1, p1 = survdiff(ti, st, gr, entry=entry)
    z2, p2 = survdiff(ti, st, gr)
    assert_allclose(z1, z2)
    assert_allclose(p1, p2)