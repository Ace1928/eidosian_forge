from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
def test_to_jsonl_count_new_lines():
    df = DataFrame([[1, 2], [1, 2]], columns=['a', 'b'])
    actual_new_lines_count = df.to_json(orient='records', lines=True).count('\n')
    expected_new_lines_count = 2
    assert actual_new_lines_count == expected_new_lines_count