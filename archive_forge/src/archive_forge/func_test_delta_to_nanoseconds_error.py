import re
import numpy as np
import pytest
from pandas._libs.tslibs.timedeltas import (
from pandas import (
import pandas._testing as tm
def test_delta_to_nanoseconds_error():
    obj = np.array([123456789], dtype='m8[ns]')
    with pytest.raises(TypeError, match="<class 'numpy.ndarray'>"):
        delta_to_nanoseconds(obj)
    with pytest.raises(TypeError, match='float'):
        delta_to_nanoseconds(1.5)
    with pytest.raises(TypeError, match='int'):
        delta_to_nanoseconds(1)
    with pytest.raises(TypeError, match='int'):
        delta_to_nanoseconds(np.int64(2))
    with pytest.raises(TypeError, match='int'):
        delta_to_nanoseconds(np.int32(3))