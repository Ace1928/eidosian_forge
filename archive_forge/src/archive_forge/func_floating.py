from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.validation import (
from statsmodels.tools.validation.validation import _right_squeeze
@pytest.fixture(params=(1.0, 1.1, np.float32(1.2), np.array([1.2]), 1.2 + 0j))
def floating(request):
    return request.param