from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_col_format_len(styler):
    result = styler.to_latex(environment='longtable', column_format='lrr{10cm}')
    expected = '\\multicolumn{4}{r}{Continued on next page} \\\\'
    assert expected in result