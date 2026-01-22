from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_multi_options(df_ext):
    cidx = MultiIndex.from_tuples([('Z', 'a'), ('Z', 'b'), ('Y', 'c')])
    ridx = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'c')])
    df_ext.index, df_ext.columns = (ridx, cidx)
    styler = df_ext.style.format(precision=2)
    expected = dedent('     &  & \\multicolumn{2}{r}{Z} & Y \\\\\n     &  & a & b & c \\\\\n    \\multirow[c]{2}{*}{A} & a & 0 & -0.61 & ab \\\\\n    ')
    result = styler.to_latex()
    assert expected in result
    with option_context('styler.latex.multicol_align', 'l'):
        assert ' &  & \\multicolumn{2}{l}{Z} & Y \\\\' in styler.to_latex()
    with option_context('styler.latex.multirow_align', 'b'):
        assert '\\multirow[b]{2}{*}{A} & a & 0 & -0.61 & ab \\\\' in styler.to_latex()