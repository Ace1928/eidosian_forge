import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_invalid_argument(slice_test_grouped):
    with pytest.raises(TypeError, match='Invalid index'):
        slice_test_grouped.nth(3.14)