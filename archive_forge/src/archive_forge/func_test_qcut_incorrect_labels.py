import os
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.tseries.offsets import Day
@pytest.mark.parametrize('labels', ['foo', 1, True])
def test_qcut_incorrect_labels(labels):
    values = range(5)
    msg = 'Bin labels must either be False, None or passed in as a list-like argument'
    with pytest.raises(ValueError, match=msg):
        qcut(values, 4, labels=labels)