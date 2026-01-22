from __future__ import annotations
import os
import pytest
from pandas.compat._optional import VERSIONS
from pandas import (
import pandas._testing as tm
@pytest.fixture(params=_utf_values)
def utf_value(request):
    """
    Fixture for all possible integer values for a UTF encoding.
    """
    return request.param