from collections import ChainMap
import inspect
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_mi(self, frame_or_series):
    obj = frame_or_series([11, 21, 31], index=MultiIndex.from_tuples([('A', x) for x in ['a', 'B', 'c']]))
    obj.rename(str.lower)