import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas._testing as tm
def test_wrong_number_names(index):
    names = index.nlevels * ['apple', 'banana', 'carrot']
    with pytest.raises(ValueError, match='^Length'):
        index.names = names