from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
@pytest.mark.parametrize('chunksize', [None, 1, 2])
def test_readjson_chunks_multiple_empty_lines(chunksize):
    j = '\n\n    {"A":1,"B":4}\n\n\n\n    {"A":2,"B":5}\n\n\n\n\n\n\n\n    {"A":3,"B":6}\n    '
    orig = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    test = read_json(StringIO(j), lines=True, chunksize=chunksize)
    if chunksize is not None:
        with test:
            test = pd.concat(test)
    tm.assert_frame_equal(orig, test, obj=f'chunksize: {chunksize}')