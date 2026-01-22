import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_datetime_no_subtract_reducelike(self):
    arr = np.array(['2021-12-02', '2019-05-12'], dtype='M8[ms]')
    msg = 'the resolved dtypes are not compatible'
    with pytest.raises(TypeError, match=msg):
        np.subtract.reduce(arr)
    with pytest.raises(TypeError, match=msg):
        np.subtract.accumulate(arr)
    with pytest.raises(TypeError, match=msg):
        np.subtract.reduceat(arr, [0])