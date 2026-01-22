from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
def test_to_timedelta_invalid_errors(self):
    msg = 'errors must be one of'
    with pytest.raises(ValueError, match=msg):
        to_timedelta(['foo'], errors='never')