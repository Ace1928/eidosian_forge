import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sort_index_preserve_levels(self, multiindex_dataframe_random_data):
    frame = multiindex_dataframe_random_data
    result = frame.sort_index()
    assert result.index.names == frame.index.names