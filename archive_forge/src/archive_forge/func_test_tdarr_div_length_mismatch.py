from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_tdarr_div_length_mismatch(self, box_with_array):
    rng = TimedeltaIndex(['1 days', NaT, '2 days'])
    mismatched = [1, 2, 3, 4]
    rng = tm.box_expected(rng, box_with_array)
    msg = 'Cannot divide vectors|Unable to coerce to Series'
    for obj in [mismatched, mismatched[:2]]:
        for other in [obj, np.array(obj), Index(obj)]:
            with pytest.raises(ValueError, match=msg):
                rng / other
            with pytest.raises(ValueError, match=msg):
                other / rng