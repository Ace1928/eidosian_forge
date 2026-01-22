from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_frame_datetime64_pre1900_repr(self):
    df = DataFrame({'year': date_range('1/1/1700', periods=50, freq='A-DEC')})
    repr(df)