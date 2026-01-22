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
def test_row_trimming_hide_index():
    df = DataFrame([[1], [2], [3], [4], [5]])
    with option_context('styler.render.max_rows', 2):
        ctx = df.style.hide([0, 1], axis='index')._translate(True, True)
    assert len(ctx['body']) == 3
    for r, val in enumerate(['3', '4', '...']):
        assert ctx['body'][r][1]['display_value'] == val