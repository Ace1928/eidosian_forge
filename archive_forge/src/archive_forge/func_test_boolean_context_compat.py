import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas._testing as tm
def test_boolean_context_compat(index):
    with pytest.raises(ValueError, match='The truth value of a'):
        if index:
            pass
    with pytest.raises(ValueError, match='The truth value of a'):
        bool(index)