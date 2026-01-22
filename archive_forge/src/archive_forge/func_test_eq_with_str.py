import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import (
def test_eq_with_str(self, dtype):
    assert dtype == dtype.name
    assert dtype != dtype.name + '-suffix'