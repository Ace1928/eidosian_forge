from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_big_name_printing(self):
    s = Series(range(1000))
    s.name = 'test'
    assert 'Name: test' in repr(s)
    s.name = None
    assert 'Name:' not in repr(s)