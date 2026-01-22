from textwrap import dedent
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.clipboard import (
def test_infer_excel_with_nulls(self, clipboard):
    text = 'col1\tcol2\n1\tred\n\tblue\n2\tgreen'
    clipboard.setText(text)
    df = read_clipboard()
    df_expected = DataFrame(data={'col1': [1, None, 2], 'col2': ['red', 'blue', 'green']})
    tm.assert_frame_equal(df, df_expected)