import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_coerce_list(self):
    arr = Index([1, 2, 3, 4])
    assert isinstance(arr, Index)
    arr = Index([1, 2, 3, 4], dtype=object)
    assert type(arr) is Index