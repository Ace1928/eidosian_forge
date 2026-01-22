from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
def test_intersection_nosort(self):
    result = Index(['c', 'b', 'a']).intersection(['b', 'a'])
    expected = Index(['b', 'a'])
    tm.assert_index_equal(result, expected)