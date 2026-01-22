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
def test_binary_mode():
    """
    read_fwf supports opening files in binary mode.

    GH 18035.
    """
    data = 'aaa aaa aaa\nbba bab b a'
    df_reference = DataFrame([['bba', 'bab', 'b a']], columns=['aaa', 'aaa.1', 'aaa.2'], index=[0])
    with tm.ensure_clean() as path:
        Path(path).write_text(data, encoding='utf-8')
        with open(path, 'rb') as file:
            df = read_fwf(file)
            file.seek(0)
            tm.assert_frame_equal(df, df_reference)