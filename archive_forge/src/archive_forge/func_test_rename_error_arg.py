from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_error_arg(self):
    ser = Series(['foo', 'bar'])
    match = re.escape('[2] not found in axis')
    with pytest.raises(KeyError, match=match):
        ser.rename({2: 9}, errors='raise')