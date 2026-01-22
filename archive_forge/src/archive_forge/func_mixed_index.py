import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.fixture
def mixed_index(self, dtype):
    return Index([1.5, 2, 3, 4, 5], dtype=dtype)