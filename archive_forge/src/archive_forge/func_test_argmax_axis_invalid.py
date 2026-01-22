import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas._testing as tm
def test_argmax_axis_invalid(self, index):
    msg = '`axis` must be fewer than the number of dimensions \\(1\\)'
    with pytest.raises(ValueError, match=msg):
        index.argmax(axis=1)
    with pytest.raises(ValueError, match=msg):
        index.argmin(axis=2)
    with pytest.raises(ValueError, match=msg):
        index.min(axis=-2)
    with pytest.raises(ValueError, match=msg):
        index.max(axis=-3)