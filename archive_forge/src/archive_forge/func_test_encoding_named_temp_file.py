from io import (
import os
import tempfile
import uuid
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_encoding_named_temp_file(all_parsers):
    parser = all_parsers
    encoding = 'shift-jis'
    title = 'てすと'
    data = 'こむ'
    expected = DataFrame({title: [data]})
    with tempfile.NamedTemporaryFile() as f:
        f.write(f'{title}\n{data}'.encode(encoding))
        f.seek(0)
        result = parser.read_csv(f, encoding=encoding)
        tm.assert_frame_equal(result, expected)
        assert not f.closed