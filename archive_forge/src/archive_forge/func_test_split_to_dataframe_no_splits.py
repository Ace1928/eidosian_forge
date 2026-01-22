from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_split_to_dataframe_no_splits(any_string_dtype):
    s = Series(['nosplit', 'alsonosplit'], dtype=any_string_dtype)
    result = s.str.split('_', expand=True)
    exp = DataFrame({0: Series(['nosplit', 'alsonosplit'], dtype=any_string_dtype)})
    tm.assert_frame_equal(result, exp)