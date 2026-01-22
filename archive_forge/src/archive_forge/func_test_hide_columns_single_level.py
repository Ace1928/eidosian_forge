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
def test_hide_columns_single_level(self, df):
    ctx = df.style._translate(True, True)
    assert ctx['head'][0][1]['is_visible']
    assert ctx['head'][0][1]['display_value'] == 'A'
    assert ctx['head'][0][2]['is_visible']
    assert ctx['head'][0][2]['display_value'] == 'B'
    assert ctx['body'][0][1]['is_visible']
    assert ctx['body'][1][2]['is_visible']
    ctx = df.style.hide('A', axis='columns')._translate(True, True)
    assert not ctx['head'][0][1]['is_visible']
    assert not ctx['body'][0][1]['is_visible']
    assert ctx['body'][1][2]['is_visible']
    ctx = df.style.hide(['A', 'B'], axis='columns')._translate(True, True)
    assert not ctx['head'][0][1]['is_visible']
    assert not ctx['head'][0][2]['is_visible']
    assert not ctx['body'][0][1]['is_visible']
    assert not ctx['body'][1][2]['is_visible']