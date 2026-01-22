from datetime import datetime
import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas import (
def test_extract_expand_True_single_capture_group_raises(index_or_series, any_string_dtype):
    s_or_idx = index_or_series(['A1', 'B2', 'C3'], dtype=any_string_dtype)
    msg = 'pattern contains no capture groups'
    with pytest.raises(ValueError, match=msg):
        s_or_idx.str.extract('[ABC][123]', expand=True)
    with pytest.raises(ValueError, match=msg):
        s_or_idx.str.extract('(?:[AB]).*', expand=True)