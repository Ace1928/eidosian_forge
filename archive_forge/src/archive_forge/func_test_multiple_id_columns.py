import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_multiple_id_columns(self):
    df = DataFrame({'famid': [1, 1, 1, 2, 2, 2, 3, 3, 3], 'birth': [1, 2, 3, 1, 2, 3, 1, 2, 3], 'ht1': [2.8, 2.9, 2.2, 2, 1.8, 1.9, 2.2, 2.3, 2.1], 'ht2': [3.4, 3.8, 2.9, 3.2, 2.8, 2.4, 3.3, 3.4, 2.9]})
    expected = DataFrame({'ht': [2.8, 3.4, 2.9, 3.8, 2.2, 2.9, 2.0, 3.2, 1.8, 2.8, 1.9, 2.4, 2.2, 3.3, 2.3, 3.4, 2.1, 2.9], 'famid': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3], 'birth': [1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3], 'age': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]})
    expected = expected.set_index(['famid', 'birth', 'age'])[['ht']]
    result = wide_to_long(df, 'ht', i=['famid', 'birth'], j='age')
    tm.assert_frame_equal(result, expected)