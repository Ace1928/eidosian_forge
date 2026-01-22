import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reflected_comparison_with_scalars(self):
    cat = Categorical([1, 2, 3], ordered=True)
    tm.assert_numpy_array_equal(cat > cat[0], np.array([False, True, True]))
    tm.assert_numpy_array_equal(cat[0] < cat, np.array([False, True, True]))