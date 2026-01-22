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
@pytest.mark.parametrize('axis', [0, 1])
def test_apply_series_return(self, axis):
    df = DataFrame([[1, 2], [3, 4]], index=['X', 'Y'], columns=['X', 'Y'])
    func = lambda s: Series(['color: red;'], index=['Y'])
    result = df.style.apply(func, axis=axis)._compute().ctx
    assert result[1, 1] == [('color', 'red')]
    assert result[1 - axis, axis] == [('color', 'red')]
    func = lambda s: Series(['color: red;', 'color: blue;'], index=['Y', 'X'])
    result = df.style.apply(func, axis=axis)._compute().ctx
    assert result[0, 0] == [('color', 'blue')]
    assert result[1, 1] == [('color', 'red')]
    assert result[1 - axis, axis] == [('color', 'red')]
    assert result[axis, 1 - axis] == [('color', 'blue')]