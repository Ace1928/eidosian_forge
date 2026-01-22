from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_repr_should_return_str(self):
    data = [8, 5, 3, 5]
    index1 = ['σ', 'τ', 'υ', 'φ']
    df = Series(data, index=index1)
    assert type(df.__repr__() == str)