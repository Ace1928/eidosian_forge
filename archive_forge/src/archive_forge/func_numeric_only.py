from datetime import (
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
@pytest.fixture(params=[True, False])
def numeric_only(request):
    """numeric_only keyword argument"""
    return request.param