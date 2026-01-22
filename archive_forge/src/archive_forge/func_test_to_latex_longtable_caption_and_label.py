import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_longtable_caption_and_label(self, df_short, caption_longtable, label_longtable):
    result = df_short.to_latex(longtable=True, caption=caption_longtable, label=label_longtable)
    expected = _dedent('\n        \\begin{longtable}{lrl}\n        \\caption{a table in a \\texttt{longtable} environment} \\label{tab:longtable} \\\\\n        \\toprule\n         & a & b \\\\\n        \\midrule\n        \\endfirsthead\n        \\caption[]{a table in a \\texttt{longtable} environment} \\\\\n        \\toprule\n         & a & b \\\\\n        \\midrule\n        \\endhead\n        \\midrule\n        \\multicolumn{3}{r}{Continued on next page} \\\\\n        \\midrule\n        \\endfoot\n        \\bottomrule\n        \\endlastfoot\n        0 & 1 & b1 \\\\\n        1 & 2 & b2 \\\\\n        \\end{longtable}\n        ')
    assert result == expected