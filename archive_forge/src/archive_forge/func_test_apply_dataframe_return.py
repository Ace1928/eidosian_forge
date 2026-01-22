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
@pytest.mark.parametrize('index', [False, True])
@pytest.mark.parametrize('columns', [False, True])
def test_apply_dataframe_return(self, index, columns):
    df = DataFrame([[1, 2], [3, 4]], index=['X', 'Y'], columns=['X', 'Y'])
    idxs = ['X', 'Y'] if index else ['Y']
    cols = ['X', 'Y'] if columns else ['Y']
    df_styles = DataFrame('color: red;', index=idxs, columns=cols)
    result = df.style.apply(lambda x: df_styles, axis=None)._compute().ctx
    assert result[1, 1] == [('color', 'red')]
    assert (result[0, 1] == [('color', 'red')]) is index
    assert (result[1, 0] == [('color', 'red')]) is columns
    assert (result[0, 0] == [('color', 'red')]) is (index and columns)