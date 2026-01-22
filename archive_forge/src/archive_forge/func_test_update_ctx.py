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
def test_update_ctx(self, styler):
    styler._update_ctx(DataFrame({'A': ['color: red', 'color: blue']}))
    expected = {(0, 0): [('color', 'red')], (1, 0): [('color', 'blue')]}
    assert styler.ctx == expected