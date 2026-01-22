import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_sample_invalid_random_state(self, obj):
    msg = 'random_state must be an integer, array-like, a BitGenerator, Generator, a numpy RandomState, or None'
    with pytest.raises(ValueError, match=msg):
        obj.sample(random_state='a_string')