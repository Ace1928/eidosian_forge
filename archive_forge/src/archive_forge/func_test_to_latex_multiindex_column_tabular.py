import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_multiindex_column_tabular(self):
    df = DataFrame({('x', 'y'): ['a']})
    result = df.to_latex()
    expected = _dedent('\n            \\begin{tabular}{ll}\n            \\toprule\n             & x \\\\\n             & y \\\\\n            \\midrule\n            0 & a \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
    assert result == expected