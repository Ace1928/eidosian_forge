import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_groupby_tabular(self):
    df = DataFrame({'a': [0, 0, 1, 1], 'b': list('abab'), 'c': [1, 2, 3, 4]})
    result = df.groupby('a').describe().to_latex(float_format='{:.1f}'.format, escape=True)
    expected = _dedent('\n            \\begin{tabular}{lrrrrrrrr}\n            \\toprule\n             & \\multicolumn{8}{r}{c} \\\\\n             & count & mean & std & min & 25\\% & 50\\% & 75\\% & max \\\\\n            a &  &  &  &  &  &  &  &  \\\\\n            \\midrule\n            0 & 2.0 & 1.5 & 0.7 & 1.0 & 1.2 & 1.5 & 1.8 & 2.0 \\\\\n            1 & 2.0 & 3.5 & 0.7 & 3.0 & 3.2 & 3.5 & 3.8 & 4.0 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
    assert result == expected