from decimal import Decimal
from io import (
import mmap
import os
import tarfile
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('malformed', ['1\r1\r1\r 1\r 1\r', '1\r1\r1\r 1\r 1\r11\r', '1\r1\r1\r 1\r 1\r11\r1\r'], ids=['words pointer', 'stream pointer', 'lines pointer'])
def test_buffer_overflow(c_parser_only, malformed):
    msg = 'Buffer overflow caught - possible malformed input file.'
    parser = c_parser_only
    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(malformed))