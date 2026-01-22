import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_multiindex_empty_name(self):
    mi = pd.MultiIndex.from_product([[1, 2]], names=[''])
    df = DataFrame(-1, index=mi, columns=range(4))
    observed = df.to_latex()
    expected = _dedent('\n            \\begin{tabular}{lrrrr}\n            \\toprule\n             & 0 & 1 & 2 & 3 \\\\\n             &  &  &  &  \\\\\n            \\midrule\n            1 & -1 & -1 & -1 & -1 \\\\\n            2 & -1 & -1 & -1 & -1 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
    assert observed == expected