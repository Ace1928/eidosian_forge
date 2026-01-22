from datetime import (
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
def test_at_applied_for_rows(self):
    df = DataFrame(index=['a'], columns=['col1', 'col2'])
    new_row = [123, 15]
    with pytest.raises(InvalidIndexError, match=f'You can only assign a scalar value not a \\{type(new_row)}'):
        df.at['a'] = new_row