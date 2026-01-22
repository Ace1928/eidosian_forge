from datetime import datetime
from inspect import signature
from io import StringIO
import os
from pathlib import Path
import sys
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
from pandas.io.parsers import TextFileReader
from pandas.io.parsers.c_parser_wrapper import CParserWrapper
@xfail_pyarrow
def test_encoding_surrogatepass(all_parsers):
    parser = all_parsers
    content = b'\xed\xbd\xbf'
    decoded = content.decode('utf-8', errors='surrogatepass')
    expected = DataFrame({decoded: [decoded]}, index=[decoded * 2])
    expected.index.name = decoded * 2
    with tm.ensure_clean() as path:
        Path(path).write_bytes(content * 2 + b',' + content + b'\n' + content * 2 + b',' + content)
        df = parser.read_csv(path, encoding_errors='surrogatepass', index_col=0)
        tm.assert_frame_equal(df, expected)
        with pytest.raises(UnicodeDecodeError, match="'utf-8' codec can't decode byte"):
            parser.read_csv(path)