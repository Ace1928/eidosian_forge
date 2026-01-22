from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
@pytest.mark.parametrize('method', ['to_string', 'to_html', 'to_latex'])
@pytest.mark.parametrize('encoding, data', [(None, 'abc'), ('utf-8', 'abc'), ('gbk', '造成输出中文显示乱码'), ('foo', 'abc')])
def test_filepath_or_buffer_arg(method, filepath_or_buffer, assert_filepath_or_buffer_equals, encoding, data, filepath_or_buffer_id):
    df = DataFrame([data])
    if method in ['to_latex']:
        pytest.importorskip('jinja2')
    if filepath_or_buffer_id not in ['string', 'pathlike'] and encoding is not None:
        with pytest.raises(ValueError, match='buf is not a file name and encoding is specified.'):
            getattr(df, method)(buf=filepath_or_buffer, encoding=encoding)
    elif encoding == 'foo':
        with pytest.raises(LookupError, match='unknown encoding'):
            getattr(df, method)(buf=filepath_or_buffer, encoding=encoding)
    else:
        expected = getattr(df, method)()
        getattr(df, method)(buf=filepath_or_buffer, encoding=encoding)
        assert_filepath_or_buffer_equals(expected)