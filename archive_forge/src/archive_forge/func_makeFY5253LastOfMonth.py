from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytest
from pandas import Timestamp
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
def makeFY5253LastOfMonth(*args, **kwds):
    return FY5253(*args, variation='last', **kwds)