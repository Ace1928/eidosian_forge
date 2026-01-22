import contextlib
import copy
import re
from textwrap import dedent
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.formats.style import (  # isort:skip
from pandas.io.formats.style_render import (
@pytest.mark.parametrize('box', [list, Series, np.array])
def test_list_slice(self, box):
    subset = box(['A'])
    df = DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['A', 'B'])
    expected = IndexSlice[:, ['A']]
    result = non_reducing_slice(subset)
    tm.assert_frame_equal(df.loc[result], df.loc[expected])