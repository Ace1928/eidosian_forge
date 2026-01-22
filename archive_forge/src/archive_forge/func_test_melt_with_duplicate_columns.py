import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_melt_with_duplicate_columns(self):
    df = DataFrame([['id', 2, 3]], columns=['a', 'b', 'b'])
    result = df.melt(id_vars=['a'], value_vars=['b'])
    expected = DataFrame([['id', 'b', 2], ['id', 'b', 3]], columns=['a', 'variable', 'value'])
    tm.assert_frame_equal(result, expected)