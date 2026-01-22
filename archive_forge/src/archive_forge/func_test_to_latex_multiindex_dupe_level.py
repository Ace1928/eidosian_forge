import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_multiindex_dupe_level(self):
    df = DataFrame(index=pd.MultiIndex.from_tuples([('A', 'c'), ('B', 'c')]), columns=['col'])
    result = df.to_latex(multirow=False)
    expected = _dedent('\n            \\begin{tabular}{lll}\n            \\toprule\n             &  & col \\\\\n            \\midrule\n            A & c & NaN \\\\\n            B & c & NaN \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
    assert result == expected