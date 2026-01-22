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
def test_row_trimming_hide_index_mi():
    df = DataFrame([[1], [2], [3], [4], [5]])
    df.index = MultiIndex.from_product([[0], [0, 1, 2, 3, 4]])
    with option_context('styler.render.max_rows', 2):
        ctx = df.style.hide([(0, 0), (0, 1)], axis='index')._translate(True, True)
    assert len(ctx['body']) == 3
    assert {'value': 0, 'attributes': 'rowspan="2"', 'is_visible': True}.items() <= ctx['body'][0][0].items()
    assert {'value': 0, 'attributes': '', 'is_visible': False}.items() <= ctx['body'][1][0].items()
    assert {'value': '...', 'is_visible': True}.items() <= ctx['body'][2][0].items()
    for r, val in enumerate(['2', '3', '...']):
        assert ctx['body'][r][1]['display_value'] == val
    for r, val in enumerate(['3', '4', '...']):
        assert ctx['body'][r][2]['display_value'] == val