import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.fixture(params=[['linear', 'single'], ['nearest', 'table']], ids=lambda x: '-'.join(x))
def interp_method(request):
    """(interpolation, method) arguments for quantile"""
    return request.param