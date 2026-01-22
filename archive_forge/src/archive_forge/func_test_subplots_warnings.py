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
@pytest.mark.slow
@pytest.mark.parametrize('idx', [range(5), date_range('1/1/2000', periods=5)])
def test_subplots_warnings(self, idx):
    with tm.assert_produces_warning(None):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), index=idx)
        df.plot(subplots=True, layout=(3, 2))