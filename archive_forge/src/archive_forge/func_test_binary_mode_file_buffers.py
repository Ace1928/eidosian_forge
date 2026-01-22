from io import (
import os
import tempfile
import uuid
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('file_path,encoding', [(('io', 'data', 'csv', 'test1.csv'), 'utf-8'), (('io', 'parser', 'data', 'unicode_series.csv'), 'latin-1'), (('io', 'parser', 'data', 'sauron.SHIFT_JIS.csv'), 'shiftjis')])
def test_binary_mode_file_buffers(all_parsers, file_path, encoding, datapath):
    parser = all_parsers
    fpath = datapath(*file_path)
    expected = parser.read_csv(fpath, encoding=encoding)
    with open(fpath, encoding=encoding) as fa:
        result = parser.read_csv(fa)
        assert not fa.closed
    tm.assert_frame_equal(expected, result)
    with open(fpath, mode='rb') as fb:
        result = parser.read_csv(fb, encoding=encoding)
        assert not fb.closed
    tm.assert_frame_equal(expected, result)
    with open(fpath, mode='rb', buffering=0) as fb:
        result = parser.read_csv(fb, encoding=encoding)
        assert not fb.closed
    tm.assert_frame_equal(expected, result)