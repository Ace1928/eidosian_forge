import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import frame_transform_kernels
from pandas.tests.frame.common import zip_frames
def test_transform_failure_valueerror():

    def op(x):
        if np.sum(np.sum(x)) < 10:
            raise ValueError
        return x
    df = DataFrame({'A': [1, 2, 3], 'B': [400, 500, 600]})
    msg = 'Transform function failed'
    with pytest.raises(ValueError, match=msg):
        df.transform([op])
    with pytest.raises(ValueError, match=msg):
        df.transform({'A': op, 'B': op})
    with pytest.raises(ValueError, match=msg):
        df.transform({'A': [op], 'B': [op]})
    with pytest.raises(ValueError, match=msg):
        df.transform({'A': [op, 'shift'], 'B': [op]})