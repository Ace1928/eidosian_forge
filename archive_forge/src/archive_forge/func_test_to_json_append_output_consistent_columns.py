from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
def test_to_json_append_output_consistent_columns():
    df1 = DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    df2 = DataFrame({'col1': [3, 4], 'col2': ['c', 'd']})
    expected = DataFrame({'col1': [1, 2, 3, 4], 'col2': ['a', 'b', 'c', 'd']})
    with tm.ensure_clean('test.json') as path:
        df1.to_json(path, lines=True, orient='records')
        df2.to_json(path, mode='a', lines=True, orient='records')
        result = read_json(path, lines=True)
        tm.assert_frame_equal(result, expected)