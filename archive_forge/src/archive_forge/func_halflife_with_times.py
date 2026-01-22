from datetime import (
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
@pytest.fixture(params=['1 day', timedelta(days=1), np.timedelta64(1, 'D')])
def halflife_with_times(request):
    """Halflife argument for EWM when times is specified."""
    return request.param