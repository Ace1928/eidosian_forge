import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
def test_preserve_metadata():
    s = Series(np.arange(100), name='foo')
    s2 = s.rolling(30).sum()
    s3 = s.rolling(20).sum()
    assert s2.name == 'foo'
    assert s3.name == 'foo'