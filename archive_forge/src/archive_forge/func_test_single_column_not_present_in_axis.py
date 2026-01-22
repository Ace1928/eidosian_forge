import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_single_column_not_present_in_axis(self):
    df = DataFrame({'A': [1, 2, 3]})
    with pytest.raises(KeyError, match="['D']"):
        df.dropna(subset='D', axis=0)