import numpy as np
from pandas import (
import pandas._testing as tm
def test_unique_data_ownership(self):
    Series(Series(['a', 'c', 'b']).unique()).sort_values()