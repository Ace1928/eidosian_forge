import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
def test_format_escape_floats(styler):
    s = styler.format('{:.1f}', escape='html')
    for expected in ['>0.0<', '>1.0<', '>-1.2<', '>-0.6<']:
        assert expected in s.to_html()
    s = styler.format(precision=1, escape='html')
    for expected in ['>0<', '>1<', '>-1.2<', '>-0.6<']:
        assert expected in s.to_html()