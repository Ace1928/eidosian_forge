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
def test_apply_none(self):

    def f(x):
        return DataFrame(np.where(x == x.max(), 'color: red', ''), index=x.index, columns=x.columns)
    result = DataFrame([[1, 2], [3, 4]]).style.apply(f, axis=None)._compute().ctx
    assert result[1, 1] == [('color', 'red')]