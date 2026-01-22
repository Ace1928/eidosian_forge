from datetime import datetime
import numpy as np
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_combine_first_name(self, datetime_series):
    result = datetime_series.combine_first(datetime_series[:5])
    assert result.name == datetime_series.name