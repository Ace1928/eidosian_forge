import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
def test_format_escape_na_rep():
    df = DataFrame([['<>&"', None]])
    s = Styler(df, uuid_len=0).format('X&{0}>X', escape='html', na_rep='&')
    ex = '<td id="T__row0_col0" class="data row0 col0" >X&&lt;&gt;&amp;&#34;>X</td>'
    expected2 = '<td id="T__row0_col1" class="data row0 col1" >&</td>'
    assert ex in s.to_html()
    assert expected2 in s.to_html()
    df = DataFrame(columns=['<>&"', None])
    styler = Styler(df, uuid_len=0)
    styler.format_index('X&{0}>X', escape='html', na_rep='&', axis=1)
    ctx = styler._translate(True, True)
    assert ctx['head'][0][1]['display_value'] == 'X&&lt;&gt;&amp;&#34;>X'
    assert ctx['head'][0][2]['display_value'] == '&'