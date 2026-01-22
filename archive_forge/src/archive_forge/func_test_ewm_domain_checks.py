import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_ewm_domain_checks():
    arr = np.random.default_rng(2).standard_normal(100)
    locs = np.arange(20, 40)
    arr[locs] = np.nan
    s = Series(arr)
    msg = 'comass must satisfy: comass >= 0'
    with pytest.raises(ValueError, match=msg):
        s.ewm(com=-0.1)
    s.ewm(com=0.0)
    s.ewm(com=0.1)
    msg = 'span must satisfy: span >= 1'
    with pytest.raises(ValueError, match=msg):
        s.ewm(span=-0.1)
    with pytest.raises(ValueError, match=msg):
        s.ewm(span=0.0)
    with pytest.raises(ValueError, match=msg):
        s.ewm(span=0.9)
    s.ewm(span=1.0)
    s.ewm(span=1.1)
    msg = 'halflife must satisfy: halflife > 0'
    with pytest.raises(ValueError, match=msg):
        s.ewm(halflife=-0.1)
    with pytest.raises(ValueError, match=msg):
        s.ewm(halflife=0.0)
    s.ewm(halflife=0.1)
    msg = 'alpha must satisfy: 0 < alpha <= 1'
    with pytest.raises(ValueError, match=msg):
        s.ewm(alpha=-0.1)
    with pytest.raises(ValueError, match=msg):
        s.ewm(alpha=0.0)
    s.ewm(alpha=0.1)
    s.ewm(alpha=1.0)
    with pytest.raises(ValueError, match=msg):
        s.ewm(alpha=1.1)