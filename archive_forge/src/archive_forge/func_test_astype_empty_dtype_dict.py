import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_empty_dtype_dict(self):
    df = DataFrame()
    result = df.astype({})
    tm.assert_frame_equal(result, df)
    assert result is not df