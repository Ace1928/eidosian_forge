import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_datetime64_block(self):
    rng = date_range('1/1/2000', periods=10)
    df = DataFrame({'time': rng})
    result = concat([df, df])
    assert (result.iloc[:10]['time'] == rng).all()
    assert (result.iloc[10:]['time'] == rng).all()