from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('name', ['', 1, 1.2, 'foo', 'αβγ', 'loooooooooooooooooooooooooooooooooooooooooooooooooooong', ('foo', 'bar', 'baz'), (1, 2), ('foo', 1, 2.3), ('α', 'β', 'γ'), ('α', 'bar')])
def test_various_names(self, name, string_series):
    string_series.name = name
    repr(string_series)