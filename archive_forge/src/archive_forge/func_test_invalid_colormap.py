import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_invalid_colormap(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)), columns=['A', 'B'])
    msg = '(is not a valid value)|(is not a known colormap)'
    with pytest.raises((ValueError, KeyError), match=msg):
        df.plot(colormap='invalid_colormap')