from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_column_format(styler):
    styler.set_table_styles([{'selector': 'column_format', 'props': ':cccc'}])
    assert '\\begin{tabular}{rrrr}' in styler.to_latex(column_format='rrrr')
    styler.set_table_styles([{'selector': 'column_format', 'props': ':r|r|cc'}])
    assert '\\begin{tabular}{r|r|cc}' in styler.to_latex()