import itertools
import sys
import pytest
import numpy as np
from numpy.testing import assert_
from scipy.special._testutils import FuncData
from scipy.special import kolmogorov, kolmogi, smirnov, smirnovi
from scipy.special._ufuncs import (_kolmogc, _kolmogci, _kolmogp,
@pytest.mark.xfail(sys.maxsize <= 2 ** 32, reason='requires 64-bit platform')
def test_oneovernclose(self):
    n = np.arange(3, 20)
    x = 1.0 / n - 2 * np.finfo(float).eps
    pp = -(n * x + 1) * (1 + x) ** (n - 2)
    dataset0 = np.column_stack([n, x, pp])
    FuncData(_smirnovp, dataset0, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
    x = 1.0 / n + 2 * np.finfo(float).eps
    pp = -(n * x + 1) * (1 + x) ** (n - 2) + 1
    dataset1 = np.column_stack([n, x, pp])
    FuncData(_smirnovp, dataset1, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])