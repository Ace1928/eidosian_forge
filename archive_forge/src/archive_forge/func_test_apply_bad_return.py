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
def test_apply_bad_return(self):

    def f(x):
        return ''
    df = DataFrame([[1, 2], [3, 4]])
    msg = 'must return a DataFrame or ndarray when passed to `Styler.apply` with axis=None'
    with pytest.raises(TypeError, match=msg):
        df.style._apply(f, axis=None)