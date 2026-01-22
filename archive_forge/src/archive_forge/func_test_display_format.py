import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
def test_display_format(styler):
    ctx = styler.format('{:0.1f}')._translate(True, True)
    assert all((['display_value' in c for c in row] for row in ctx['body']))
    assert all(([len(c['display_value']) <= 3 for c in row[1:]] for row in ctx['body']))
    assert len(ctx['body'][0][1]['display_value'].lstrip('-')) <= 3