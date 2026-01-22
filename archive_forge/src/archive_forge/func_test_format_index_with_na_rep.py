import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
def test_format_index_with_na_rep():
    df = DataFrame([[1, 2, 3, 4, 5]], columns=['A', None, np.nan, NaT, NA])
    ctx = df.style.format_index(None, na_rep='--', axis=1)._translate(True, True)
    assert ctx['head'][0][1]['display_value'] == 'A'
    for i in [2, 3, 4, 5]:
        assert ctx['head'][0][i]['display_value'] == '--'