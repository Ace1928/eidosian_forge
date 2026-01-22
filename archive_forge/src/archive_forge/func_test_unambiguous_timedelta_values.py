from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
@pytest.mark.parametrize('val, errors', [('1M', True), ('1 M', True), ('1Y', True), ('1 Y', True), ('1y', True), ('1 y', True), ('1m', False), ('1 m', False), ('1 day', False), ('2day', False)])
def test_unambiguous_timedelta_values(self, val, errors):
    msg = "Units 'M', 'Y' and 'y' do not represent unambiguous timedelta"
    if errors:
        with pytest.raises(ValueError, match=msg):
            to_timedelta(val)
    else:
        to_timedelta(val)