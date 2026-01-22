import datetime as dt
from datetime import date
import re
import numpy as np
import pytest
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_append_nondatetimeindex(self):
    rng = date_range('1/1/2000', periods=10)
    idx = Index(['a', 'b', 'c', 'd'])
    result = rng.append(idx)
    assert isinstance(result[0], Timestamp)