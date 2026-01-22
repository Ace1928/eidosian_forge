import numpy as np
from numpy.testing import assert_, assert_allclose
from numpy import pi
import pytest
import itertools
from scipy._lib import _pep440
import scipy.special as sc
from scipy.special._testutils import (
from scipy.special._mptestutils import (
from scipy.special._ufuncs import (
@check_version(mpmath, '0.13')
def test_hyp2f1_real_some():
    dataset = []
    for a in [-10, -5, -1.8, 1.8, 5, 10]:
        for b in [-2.5, -1, 1, 7.4]:
            for c in [-9, -1.8, 5, 20.4]:
                for z in [-10, -1.01, -0.99, 0, 0.6, 0.95, 1.5, 10]:
                    try:
                        v = float(mpmath.hyp2f1(a, b, c, z))
                    except Exception:
                        continue
                    dataset.append((a, b, c, z, v))
    dataset = np.array(dataset, dtype=np.float64)
    with np.errstate(invalid='ignore'):
        FuncData(sc.hyp2f1, dataset, (0, 1, 2, 3), 4, rtol=1e-09, ignore_inf_sign=True).check()