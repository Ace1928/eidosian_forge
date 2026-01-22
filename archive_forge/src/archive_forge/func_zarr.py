import re
import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.fixture
def zarr():
    """Fixture returning SparseArray with integer entries and 'fill_value=0'"""
    return SparseArray([0, 0, 1, 2, 3, 0, 4, 5, 0, 6], fill_value=0)