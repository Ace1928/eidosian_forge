import pytest
import pandas as pd
from pandas import MultiIndex
import pandas._testing as tm
def test_duplicate_level_names_access_raises(idx):
    idx.names = ['foo', 'foo']
    with pytest.raises(ValueError, match='name foo occurs multiple times'):
        idx._get_level_number('foo')