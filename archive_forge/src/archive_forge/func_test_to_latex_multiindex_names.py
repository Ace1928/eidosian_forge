import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('name0', [None, 'named0'])
@pytest.mark.parametrize('name1', [None, 'named1'])
@pytest.mark.parametrize('axes', [[0], [1], [0, 1]])
def test_to_latex_multiindex_names(self, name0, name1, axes):
    names = [name0, name1]
    mi = pd.MultiIndex.from_product([[1, 2], [3, 4]])
    df = DataFrame(-1, index=mi.copy(), columns=mi.copy())
    for idx in axes:
        df.axes[idx].names = names
    idx_names = tuple((n or '' for n in names))
    idx_names_row = f'{idx_names[0]} & {idx_names[1]} &  &  &  &  \\\\\n' if 0 in axes and any(names) else ''
    col_names = [n if bool(n) and 1 in axes else '' for n in names]
    observed = df.to_latex(multirow=False)
    expected = '\\begin{tabular}{llrrrr}\n\\toprule\n & %s & \\multicolumn{2}{r}{1} & \\multicolumn{2}{r}{2} \\\\\n & %s & 3 & 4 & 3 & 4 \\\\\n%s\\midrule\n1 & 3 & -1 & -1 & -1 & -1 \\\\\n & 4 & -1 & -1 & -1 & -1 \\\\\n2 & 3 & -1 & -1 & -1 & -1 \\\\\n & 4 & -1 & -1 & -1 & -1 \\\\\n\\bottomrule\n\\end{tabular}\n' % tuple(list(col_names) + [idx_names_row])
    assert observed == expected