import itertools
import sys
import pytest
import numpy as np
from numpy.testing import assert_
from scipy.special._testutils import FuncData
from scipy.special import kolmogorov, kolmogi, smirnov, smirnovi
from scipy.special._ufuncs import (_kolmogc, _kolmogci, _kolmogp,
def test_oneovern(self):
    n = 2 ** np.arange(1, 10)
    x = 1.0 / n
    pp = -(n * x + 1) * (1 + x) ** (n - 2) + 0.5
    dataset0 = np.column_stack([n, x, pp])
    FuncData(_smirnovp, dataset0, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])