from textwrap import dedent
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_concat_recursion(styler):
    df = styler.data
    styler1 = styler
    styler2 = Styler(df.agg(['sum']), uuid_len=0, precision=3)
    styler3 = Styler(df.agg(['sum']), uuid_len=0, precision=4)
    result = styler1.concat(styler2.concat(styler3)).to_string()
    expected = dedent('     A B C\n    0 0 -0.61 ab\n    1 1 -1.22 cd\n    sum 1 -1.830 abcd\n    sum 1 -1.8300 abcd\n    ')
    assert result == expected