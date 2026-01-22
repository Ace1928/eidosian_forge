import re
import warnings
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
@pytest.mark.parametrize('string, expected', [('Sparse[int]', 'int'), ('Sparse[int, 0]', 'int'), ('Sparse[int64]', 'int64'), ('Sparse[int64, 0]', 'int64'), ('Sparse[datetime64[ns], 0]', 'datetime64[ns]')])
def test_parse_subtype(string, expected):
    subtype, _ = SparseDtype._parse_subtype(string)
    assert subtype == expected