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
@pytest.mark.parametrize('kind, kwargs', [['bar', {'stacked': True}], ['barh', {'stacked': False}], ['barh', {'stacked': True}], ['bar', {'subplots': True}], ['barh', {'subplots': True}]])
def test_bar_barwidth_position_int_width_1(self, kind, kwargs):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    self._check_bar_alignment(df, kind=kind, width=1, **kwargs)