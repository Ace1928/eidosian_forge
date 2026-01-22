import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_multiindex_multirow(self):
    mi = pd.MultiIndex.from_product([[0.0, 1.0], [3.0, 2.0, 1.0], ['0', '1']], names=['i', 'val0', 'val1'])
    df = DataFrame(index=mi)
    result = df.to_latex(multirow=True, escape=False)
    expected = _dedent('\n            \\begin{tabular}{lll}\n            \\toprule\n            i & val0 & val1 \\\\\n            \\midrule\n            \\multirow[t]{6}{*}{0.000000} & \\multirow[t]{2}{*}{3.000000} & 0 \\\\\n             &  & 1 \\\\\n            \\cline{2-3}\n             & \\multirow[t]{2}{*}{2.000000} & 0 \\\\\n             &  & 1 \\\\\n            \\cline{2-3}\n             & \\multirow[t]{2}{*}{1.000000} & 0 \\\\\n             &  & 1 \\\\\n            \\cline{1-3} \\cline{2-3}\n            \\multirow[t]{6}{*}{1.000000} & \\multirow[t]{2}{*}{3.000000} & 0 \\\\\n             &  & 1 \\\\\n            \\cline{2-3}\n             & \\multirow[t]{2}{*}{2.000000} & 0 \\\\\n             &  & 1 \\\\\n            \\cline{2-3}\n             & \\multirow[t]{2}{*}{1.000000} & 0 \\\\\n             &  & 1 \\\\\n            \\cline{1-3} \\cline{2-3}\n            \\bottomrule\n            \\end{tabular}\n            ')
    assert result == expected