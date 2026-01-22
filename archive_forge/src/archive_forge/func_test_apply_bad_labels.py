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
@pytest.mark.parametrize('axis', ['index', 'columns'])
def test_apply_bad_labels(self, axis):

    def f(x):
        return DataFrame(**{axis: ['bad', 'labels']})
    df = DataFrame([[1, 2], [3, 4]])
    msg = f'created invalid {axis} labels.'
    with pytest.raises(ValueError, match=msg):
        df.style._apply(f, axis=None)