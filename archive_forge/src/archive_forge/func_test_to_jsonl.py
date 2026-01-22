from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
def test_to_jsonl():
    df = DataFrame([[1, 2], [1, 2]], columns=['a', 'b'])
    result = df.to_json(orient='records', lines=True)
    expected = '{"a":1,"b":2}\n{"a":1,"b":2}\n'
    assert result == expected
    df = DataFrame([['foo}', 'bar'], ['foo"', 'bar']], columns=['a', 'b'])
    result = df.to_json(orient='records', lines=True)
    expected = '{"a":"foo}","b":"bar"}\n{"a":"foo\\"","b":"bar"}\n'
    assert result == expected
    tm.assert_frame_equal(read_json(StringIO(result), lines=True), df)
    df = DataFrame([['foo\\', 'bar'], ['foo"', 'bar']], columns=['a\\', 'b'])
    result = df.to_json(orient='records', lines=True)
    expected = '{"a\\\\":"foo\\\\","b":"bar"}\n{"a\\\\":"foo\\"","b":"bar"}\n'
    assert result == expected
    tm.assert_frame_equal(read_json(StringIO(result), lines=True), df)