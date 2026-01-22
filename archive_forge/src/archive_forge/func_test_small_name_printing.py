from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_small_name_printing(self):
    s = Series([0, 1, 2])
    s.name = 'test'
    assert 'Name: test' in repr(s)
    s.name = None
    assert 'Name:' not in repr(s)