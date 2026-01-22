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
@pytest.mark.parametrize('kwargs', [{'kind': 'bar', 'stacked': True}, {'kind': 'bar', 'stacked': True, 'width': 0.9}, {'kind': 'barh', 'stacked': True}, {'kind': 'barh', 'stacked': True, 'width': 0.9}, {'kind': 'bar', 'stacked': False}, {'kind': 'bar', 'stacked': False, 'width': 0.9}, {'kind': 'barh', 'stacked': False}, {'kind': 'barh', 'stacked': False, 'width': 0.9}, {'kind': 'bar', 'subplots': True}, {'kind': 'bar', 'subplots': True, 'width': 0.9}, {'kind': 'barh', 'subplots': True}, {'kind': 'barh', 'subplots': True, 'width': 0.9}, {'kind': 'bar', 'stacked': True, 'align': 'edge'}, {'kind': 'bar', 'stacked': True, 'width': 0.9, 'align': 'edge'}, {'kind': 'barh', 'stacked': True, 'align': 'edge'}, {'kind': 'barh', 'stacked': True, 'width': 0.9, 'align': 'edge'}, {'kind': 'bar', 'stacked': False, 'align': 'edge'}, {'kind': 'bar', 'stacked': False, 'width': 0.9, 'align': 'edge'}, {'kind': 'barh', 'stacked': False, 'align': 'edge'}, {'kind': 'barh', 'stacked': False, 'width': 0.9, 'align': 'edge'}, {'kind': 'bar', 'subplots': True, 'align': 'edge'}, {'kind': 'bar', 'subplots': True, 'width': 0.9, 'align': 'edge'}, {'kind': 'barh', 'subplots': True, 'align': 'edge'}, {'kind': 'barh', 'subplots': True, 'width': 0.9, 'align': 'edge'}])
def test_bar_align_multiple_columns(self, kwargs):
    df = DataFrame({'A': [3] * 5, 'B': list(range(5))}, index=range(5))
    self._check_bar_alignment(df, **kwargs)