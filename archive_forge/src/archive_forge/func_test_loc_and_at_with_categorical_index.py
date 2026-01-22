import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_and_at_with_categorical_index(self):
    df = DataFrame([[1, 2], [3, 4], [5, 6]], index=CategoricalIndex(['A', 'B', 'C']))
    s = df[0]
    assert s.loc['A'] == 1
    assert s.at['A'] == 1
    assert df.loc['B', 1] == 4
    assert df.at['B', 1] == 4