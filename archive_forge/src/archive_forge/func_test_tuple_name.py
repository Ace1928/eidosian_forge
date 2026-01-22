from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_tuple_name(self):
    biggie = Series(np.random.randn(1000), index=np.arange(1000), name=('foo', 'bar', 'baz'))
    repr(biggie)