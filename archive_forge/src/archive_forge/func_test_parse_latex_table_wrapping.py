from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_parse_latex_table_wrapping(styler):
    styler.set_table_styles([{'selector': 'toprule', 'props': ':value'}, {'selector': 'bottomrule', 'props': ':value'}, {'selector': 'midrule', 'props': ':value'}, {'selector': 'column_format', 'props': ':value'}])
    assert _parse_latex_table_wrapping(styler.table_styles, styler.caption) is False
    assert _parse_latex_table_wrapping(styler.table_styles, 'some caption') is True
    styler.set_table_styles([{'selector': 'not-ignored', 'props': ':value'}], overwrite=False)
    assert _parse_latex_table_wrapping(styler.table_styles, None) is True