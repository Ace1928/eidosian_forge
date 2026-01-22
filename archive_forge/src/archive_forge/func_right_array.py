import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.fixture
def right_array():
    """Fixture returning boolean array with valid and missing values."""
    return pd.array([True, False, None] * 3, dtype='boolean')