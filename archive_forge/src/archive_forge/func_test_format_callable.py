import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
def test_format_callable(styler):
    ctx = styler.format(lambda v: 'neg' if v < 0 else 'pos')._translate(True, True)
    assert ctx['body'][0][1]['display_value'] == 'pos'
    assert ctx['body'][0][2]['display_value'] == 'neg'
    assert ctx['body'][1][1]['display_value'] == 'pos'
    assert ctx['body'][1][2]['display_value'] == 'neg'