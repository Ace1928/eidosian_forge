from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.validation import (
from statsmodels.tools.validation.validation import _right_squeeze
@pytest.fixture(params=(dict, OrderedDict, CustomDict, None))
def dict_type(request):
    return request.param