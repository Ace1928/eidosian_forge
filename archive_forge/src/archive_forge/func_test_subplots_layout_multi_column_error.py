import string
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_subplots_layout_multi_column_error(self):
    df = DataFrame(np.random.default_rng(2).random((10, 3)), index=list(string.ascii_letters[:10]))
    msg = 'Layout of 1x1 must be larger than required size 3'
    with pytest.raises(ValueError, match=msg):
        df.plot(subplots=True, layout=(1, 1))
    msg = 'At least one dimension of layout must be positive'
    with pytest.raises(ValueError, match=msg):
        df.plot(subplots=True, layout=(-1, -1))