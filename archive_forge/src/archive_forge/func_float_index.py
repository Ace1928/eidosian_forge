import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.fixture
def float_index(self, dtype):
    return Index([0.0, 2.5, 5.0, 7.5, 10.0], dtype=dtype)