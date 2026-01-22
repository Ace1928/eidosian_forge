from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_axis_supported(self):
    ser = Series(range(5))
    ser.rename({}, axis=0)
    ser.rename({}, axis='index')
    with pytest.raises(ValueError, match='No axis named 5'):
        ser.rename({}, axis=5)