from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_latex_hiding_index_columns_multiindex_alignment():
    midx = MultiIndex.from_product([['i0', 'j0'], ['i1'], ['i2', 'j2']], names=['i-0', 'i-1', 'i-2'])
    cidx = MultiIndex.from_product([['c0'], ['c1', 'd1'], ['c2', 'd2']], names=['c-0', 'c-1', 'c-2'])
    df = DataFrame(np.arange(16).reshape(4, 4), index=midx, columns=cidx)
    styler = Styler(df, uuid_len=0)
    styler.hide(level=1, axis=0).hide(level=0, axis=1)
    styler.hide([('i0', 'i1', 'i2')], axis=0)
    styler.hide([('c0', 'c1', 'c2')], axis=1)
    styler.map(lambda x: 'color:{red};' if x == 5 else '')
    styler.map_index(lambda x: 'color:{blue};' if 'j' in x else '')
    result = styler.to_latex()
    expected = dedent('        \\begin{tabular}{llrrr}\n         & c-1 & c1 & \\multicolumn{2}{r}{d1} \\\\\n         & c-2 & d2 & c2 & d2 \\\\\n        i-0 & i-2 &  &  &  \\\\\n        i0 & \\color{blue} j2 & \\color{red} 5 & 6 & 7 \\\\\n        \\multirow[c]{2}{*}{\\color{blue} j0} & i2 & 9 & 10 & 11 \\\\\n        \\color{blue}  & \\color{blue} j2 & 13 & 14 & 15 \\\\\n        \\end{tabular}\n        ')
    assert result == expected