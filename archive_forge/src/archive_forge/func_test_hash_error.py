import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas._testing as tm
def test_hash_error(index):
    with pytest.raises(TypeError, match=f"unhashable type: '{type(index).__name__}'"):
        hash(index)