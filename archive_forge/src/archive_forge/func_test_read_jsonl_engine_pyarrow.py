from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
def test_read_jsonl_engine_pyarrow(datapath, engine):
    result = read_json(datapath('io', 'json', 'data', 'line_delimited.json'), lines=True, engine=engine)
    expected = DataFrame({'a': [1, 3, 5], 'b': [2, 4, 6]})
    tm.assert_frame_equal(result, expected)