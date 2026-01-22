from datetime import datetime
from itertools import permutations
import numpy as np
from pandas._libs import algos as libalgos
import pandas._testing as tm
def test_infinity_against_nan(self):
    Inf = libalgos.Infinity()
    NegInf = libalgos.NegInfinity()
    assert not Inf > np.nan
    assert not Inf >= np.nan
    assert not Inf < np.nan
    assert not Inf <= np.nan
    assert not Inf == np.nan
    assert Inf != np.nan
    assert not NegInf > np.nan
    assert not NegInf >= np.nan
    assert not NegInf < np.nan
    assert not NegInf <= np.nan
    assert not NegInf == np.nan
    assert NegInf != np.nan