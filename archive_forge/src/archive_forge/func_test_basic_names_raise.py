from io import StringIO
import pytest
from pandas import DataFrame
import pandas._testing as tm
def test_basic_names_raise(all_parsers):
    parser = all_parsers
    data = '0,1,2\n3,4,5'
    with pytest.raises(ValueError, match='Duplicate names'):
        parser.read_csv(StringIO(data), names=['a', 'b', 'a'])