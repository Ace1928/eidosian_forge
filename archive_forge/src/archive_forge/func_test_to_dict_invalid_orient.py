from collections import (
from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
def test_to_dict_invalid_orient(self):
    df = DataFrame({'A': [0, 1]})
    msg = "orient 'xinvalid' not understood"
    with pytest.raises(ValueError, match=msg):
        df.to_dict(orient='xinvalid')