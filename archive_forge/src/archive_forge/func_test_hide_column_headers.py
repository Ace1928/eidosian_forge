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
def test_hide_column_headers(self, df, styler):
    ctx = styler.hide(axis='columns')._translate(True, True)
    assert len(ctx['head']) == 0
    df.index.name = 'some_name'
    ctx = df.style.hide(axis='columns')._translate(True, True)
    assert len(ctx['head']) == 1