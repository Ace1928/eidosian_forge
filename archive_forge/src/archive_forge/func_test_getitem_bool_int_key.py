from datetime import timedelta
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
from pandas import (
import pandas._testing as tm
def test_getitem_bool_int_key():
    ser = Series({True: 1, False: 0})
    with pytest.raises(KeyError, match='0'):
        ser.loc[0]