import csv
from io import StringIO
import pytest
from pandas.compat import PY311
from pandas.errors import ParserError
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('quote_char', ['~', '*', '%', '$', '@', 'P'])
def test_quote_char_various(all_parsers, quote_char):
    parser = all_parsers
    expected = DataFrame([[1, 2, 'cat']], columns=['a', 'b', 'c'])
    data = 'a,b,c\n1,2,"cat"'
    new_data = data.replace('"', quote_char)
    result = parser.read_csv(StringIO(new_data), quotechar=quote_char)
    tm.assert_frame_equal(result, expected)