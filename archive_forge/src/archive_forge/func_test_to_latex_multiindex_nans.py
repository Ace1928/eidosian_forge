import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('one_row', [True, False])
def test_to_latex_multiindex_nans(self, one_row):
    df = DataFrame({'a': [None, 1], 'b': [2, 3], 'c': [4, 5]})
    if one_row:
        df = df.iloc[[0]]
    observed = df.set_index(['a', 'b']).to_latex(multirow=False)
    expected = _dedent('\n            \\begin{tabular}{llr}\n            \\toprule\n             &  & c \\\\\n            a & b &  \\\\\n            \\midrule\n            NaN & 2 & 4 \\\\\n            ')
    if not one_row:
        expected += '1.000000 & 3 & 5 \\\\\n'
    expected += '\\bottomrule\n\\end{tabular}\n'
    assert observed == expected