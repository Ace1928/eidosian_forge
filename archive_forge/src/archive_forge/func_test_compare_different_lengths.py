import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_compare_different_lengths(self):
    c1 = Categorical([], categories=['a', 'b'])
    c2 = Categorical([], categories=['a'])
    msg = "Categoricals can only be compared if 'categories' are the same."
    with pytest.raises(TypeError, match=msg):
        c1 == c2