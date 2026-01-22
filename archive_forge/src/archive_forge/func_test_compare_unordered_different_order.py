import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_compare_unordered_different_order(self):
    a = Categorical(['a'], categories=['a', 'b'])
    b = Categorical(['b'], categories=['b', 'a'])
    assert not a.equals(b)