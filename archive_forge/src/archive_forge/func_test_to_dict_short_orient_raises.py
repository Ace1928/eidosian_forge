from collections import (
from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('orient', ['d', 'l', 'r', 'sp', 's', 'i'])
def test_to_dict_short_orient_raises(self, orient):
    df = DataFrame({'A': [0, 1]})
    with pytest.raises(ValueError, match='not understood'):
        df.to_dict(orient=orient)