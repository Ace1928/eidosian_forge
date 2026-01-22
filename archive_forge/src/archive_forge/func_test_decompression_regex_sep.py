from __future__ import annotations
import csv
from io import (
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('compression,klass', [('gzip', 'GzipFile'), ('bz2', 'BZ2File')])
def test_decompression_regex_sep(python_parser_only, csv1, compression, klass):
    parser = python_parser_only
    with open(csv1, 'rb') as f:
        data = f.read()
    data = data.replace(b',', b'::')
    expected = parser.read_csv(csv1)
    module = pytest.importorskip(compression)
    klass = getattr(module, klass)
    with tm.ensure_clean() as path:
        with klass(path, mode='wb') as tmp:
            tmp.write(data)
        result = parser.read_csv(path, sep='::', compression=compression)
        tm.assert_frame_equal(result, expected)