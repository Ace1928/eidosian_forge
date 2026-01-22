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
@pytest.mark.parametrize('filename', ['sé-es-vé.csv', 'ru-sй.csv', '中文文件名.csv'])
def test_filename_with_special_chars(all_parsers, filename):
    parser = all_parsers
    df = DataFrame({'a': [1, 2, 3]})
    with tm.ensure_clean(filename) as path:
        df.to_csv(path, index=False)
        result = parser.read_csv(path)
        tm.assert_frame_equal(result, df)