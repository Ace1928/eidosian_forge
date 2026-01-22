import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_multiindex_tabular(self, multiindex_frame):
    result = multiindex_frame.to_latex(multirow=False)
    expected = _dedent('\n            \\begin{tabular}{llrrrr}\n            \\toprule\n             &  & 0 & 1 & 2 & 3 \\\\\n            \\midrule\n            c1 & 0 & 0 & 1 & 2 & 3 \\\\\n             & 1 & 4 & 5 & 6 & 7 \\\\\n            c2 & 0 & 0 & 1 & 2 & 3 \\\\\n             & 1 & 4 & 5 & 6 & 7 \\\\\n            c3 & 0 & 0 & 1 & 2 & 3 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
    assert result == expected