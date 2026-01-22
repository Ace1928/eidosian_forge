from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
def test_readjson_chunksize_requires_lines(lines_json_df, engine):
    msg = 'chunksize can only be passed if lines=True'
    with pytest.raises(ValueError, match=msg):
        with read_json(StringIO(lines_json_df), lines=False, chunksize=2, engine=engine) as _:
            pass