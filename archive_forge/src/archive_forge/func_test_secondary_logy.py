from datetime import datetime
from itertools import chain
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.parametrize('input_logy, expected_scale', [(True, 'log'), ('sym', 'symlog')])
def test_secondary_logy(self, input_logy, expected_scale):
    s1 = Series(np.random.default_rng(2).standard_normal(100))
    s2 = Series(np.random.default_rng(2).standard_normal(100))
    ax1 = s1.plot(logy=input_logy)
    ax2 = s2.plot(secondary_y=True, logy=input_logy)
    assert ax1.get_yscale() == expected_scale
    assert ax2.get_yscale() == expected_scale