from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.validation import (
from statsmodels.tools.validation.validation import _right_squeeze
@pytest.fixture(params=(np.empty(2), 1.2 + 1j, True, '3.2', None))
def not_floating(request):
    return request.param