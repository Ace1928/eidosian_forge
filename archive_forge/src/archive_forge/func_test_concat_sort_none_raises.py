import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
def test_concat_sort_none_raises(self):
    df = DataFrame({1: [1, 2], 'a': [3, 4]})
    msg = "The 'sort' keyword only accepts boolean values; None was passed."
    with pytest.raises(ValueError, match=msg):
        pd.concat([df, df], sort=None)