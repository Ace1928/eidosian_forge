import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import (
def test_get_common_dtype(self, dtype):
    assert dtype._get_common_dtype([dtype]) == dtype