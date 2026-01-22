import codecs
import csv
from io import StringIO
import os
from pathlib import Path
import numpy as np
import pytest
from pandas.compat import PY311
from pandas.errors import (
from pandas import DataFrame
import pandas._testing as tm
def test_null_byte_char(request, all_parsers):
    data = '\x00,foo'
    names = ['a', 'b']
    parser = all_parsers
    if parser.engine == 'c' or (parser.engine == 'python' and PY311):
        if parser.engine == 'python' and PY311:
            request.applymarker(pytest.mark.xfail(reason='In Python 3.11, this is read as an empty character not null'))
        expected = DataFrame([[np.nan, 'foo']], columns=names)
        out = parser.read_csv(StringIO(data), names=names)
        tm.assert_frame_equal(out, expected)
    else:
        if parser.engine == 'pyarrow':
            pytest.skip(reason='https://github.com/apache/arrow/issues/38676')
        else:
            msg = 'NULL byte detected'
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), names=names)