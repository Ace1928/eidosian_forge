import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_required_arguments2(self):
    start = Period('02-Apr-2005', 'D')
    msg = 'Of the three parameters: start, end, and periods, exactly two must be specified'
    with pytest.raises(ValueError, match=msg):
        period_range(start=start)