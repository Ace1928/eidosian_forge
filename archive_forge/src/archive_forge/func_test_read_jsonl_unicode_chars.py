from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
def test_read_jsonl_unicode_chars():
    json = '{"a": "foo”", "b": "bar"}\n{"a": "foo", "b": "bar"}\n'
    json = StringIO(json)
    result = read_json(json, lines=True)
    expected = DataFrame([['foo”', 'bar'], ['foo', 'bar']], columns=['a', 'b'])
    tm.assert_frame_equal(result, expected)
    json = '{"a": "foo”", "b": "bar"}\n{"a": "foo", "b": "bar"}\n'
    result = read_json(StringIO(json), lines=True)
    expected = DataFrame([['foo”', 'bar'], ['foo', 'bar']], columns=['a', 'b'])
    tm.assert_frame_equal(result, expected)