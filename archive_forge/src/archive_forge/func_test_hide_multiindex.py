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
def test_hide_multiindex(self):
    df = DataFrame({'A': [1, 2], 'B': [1, 2]}, index=MultiIndex.from_arrays([['a', 'a'], [0, 1]], names=['idx_level_0', 'idx_level_1']))
    ctx1 = df.style._translate(True, True)
    assert ctx1['body'][0][0]['is_visible']
    assert ctx1['body'][0][1]['is_visible']
    assert len(ctx1['head'][0]) == 4
    ctx2 = df.style.hide(axis='index')._translate(True, True)
    assert not ctx2['body'][0][0]['is_visible']
    assert not ctx2['body'][0][1]['is_visible']
    assert len(ctx2['head'][0]) == 3
    assert not ctx2['head'][0][0]['is_visible']