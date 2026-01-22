from textwrap import dedent
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_format_non_unique(df):
    html = df.style.format({'d': '{:.1f}'}).to_html()
    for val in ['1.000000<', '4.000000<', '7.000000<']:
        assert val in html
    for val in ['2.0<', '3.0<', '5.0<', '6.0<', '8.0<', '9.0<']:
        assert val in html
    html = df.style.format(precision=1, subset=IndexSlice['j', 'd']).to_html()
    for val in ['1.000000<', '4.000000<', '7.000000<', '2.000000<', '3.000000<']:
        assert val in html
    for val in ['5.0<', '6.0<', '8.0<', '9.0<']:
        assert val in html