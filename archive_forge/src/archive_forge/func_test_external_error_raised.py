import os
import pytest
from pandas import (
import pandas._testing as tm
def test_external_error_raised():
    with tm.external_error_raised(TypeError):
        raise TypeError('Should not check this error message, so it will pass')