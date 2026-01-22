import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_negative_step(slice_test_grouped):
    with pytest.raises(ValueError, match='Invalid step'):
        slice_test_grouped.nth(slice(None, None, -1))