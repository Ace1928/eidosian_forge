from io import StringIO
import pytest
from pandas import DataFrame
import pandas._testing as tm
@xfail_pyarrow
@pytest.mark.parametrize('data,expected', [('a,a,a.1\n1,2,3', DataFrame([[1, 2, 3]], columns=['a', 'a.2', 'a.1'])), ('a,a,a.1,a.1.1,a.1.1.1,a.1.1.1.1\n1,2,3,4,5,6', DataFrame([[1, 2, 3, 4, 5, 6]], columns=['a', 'a.2', 'a.1', 'a.1.1', 'a.1.1.1', 'a.1.1.1.1'])), ('a,a,a.3,a.1,a.2,a,a\n1,2,3,4,5,6,7', DataFrame([[1, 2, 3, 4, 5, 6, 7]], columns=['a', 'a.4', 'a.3', 'a.1', 'a.2', 'a.5', 'a.6']))])
def test_thorough_mangle_columns(all_parsers, data, expected):
    parser = all_parsers
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)