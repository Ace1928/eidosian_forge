from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_alignment_doesnt_change_tz(self):
    dti = date_range('2016-01-01', periods=10, tz='CET')
    dti_utc = dti.tz_convert('UTC')
    ser = Series(10, index=dti)
    ser_utc = Series(10, index=dti_utc)
    ser * ser_utc
    assert ser.index is dti
    assert ser_utc.index is dti_utc