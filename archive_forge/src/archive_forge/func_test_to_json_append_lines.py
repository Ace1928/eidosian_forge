from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
def test_to_json_append_lines():
    df = DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    msg = "mode='a' \\(append\\) is only supported when lines is True and orient is 'records'"
    with pytest.raises(ValueError, match=msg):
        df.to_json(mode='a', lines=False, orient='records')