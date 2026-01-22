from datetime import datetime
from io import (
from pathlib import Path
import numpy as np
import pytest
from pandas.errors import EmptyDataError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.common import urlopen
from pandas.io.parsers import (
def test_fwf_for_uint8():
    data = '1421302965.213420    PRI=3 PGN=0xef00      DST=0x17 SRC=0x28    04 154 00 00 00 00 00 127\n1421302964.226776    PRI=6 PGN=0xf002               SRC=0x47    243 00 00 255 247 00 00 71'
    df = read_fwf(StringIO(data), colspecs=[(0, 17), (25, 26), (33, 37), (49, 51), (58, 62), (63, 1000)], names=['time', 'pri', 'pgn', 'dst', 'src', 'data'], converters={'pgn': lambda x: int(x, 16), 'src': lambda x: int(x, 16), 'dst': lambda x: int(x, 16), 'data': lambda x: len(x.split(' '))})
    expected = DataFrame([[1421302965.21342, 3, 61184, 23, 40, 8], [1421302964.226776, 6, 61442, None, 71, 8]], columns=['time', 'pri', 'pgn', 'dst', 'src', 'data'])
    expected['dst'] = expected['dst'].astype(object)
    tm.assert_frame_equal(df, expected)