import numpy as np
import pytest
from pandas import Series
import pandas._testing as tm
def no_nans(x):
    return x.notna().all().all()