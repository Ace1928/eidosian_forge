import csv
from io import StringIO
import os
import numpy as np
import pytest
from pandas.errors import ParserError
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
from pandas.io.common import get_handle
def test_to_csv_headers(self):
    from_df = DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
    to_df = DataFrame([[1, 2], [3, 4]], columns=['X', 'Y'])
    with tm.ensure_clean('__tmp_to_csv_headers__') as path:
        from_df.to_csv(path, header=['X', 'Y'])
        recons = self.read_csv(path)
        tm.assert_frame_equal(to_df, recons)
        from_df.to_csv(path, index=False, header=['X', 'Y'])
        recons = self.read_csv(path)
        return_value = recons.reset_index(inplace=True)
        assert return_value is None
        tm.assert_frame_equal(to_df, recons)