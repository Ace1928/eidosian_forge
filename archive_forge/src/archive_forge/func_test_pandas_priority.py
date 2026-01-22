import array
import subprocess
import sys
import numpy as np
import pytest
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_pandas_priority():

    class MyClass:
        __pandas_priority__ = 5000

        def __radd__(self, other):
            return self
    left = MyClass()
    right = Series(range(3))
    assert right.__add__(left) is NotImplemented
    assert right + left is left