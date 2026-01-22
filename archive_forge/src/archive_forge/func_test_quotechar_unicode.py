import csv
from io import StringIO
import pytest
from pandas.compat import PY311
from pandas.errors import ParserError
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('quotechar', ['"', '\x01'])
def test_quotechar_unicode(all_parsers, quotechar):
    data = 'a\n1'
    parser = all_parsers
    expected = DataFrame({'a': [1]})
    result = parser.read_csv(StringIO(data), quotechar=quotechar)
    tm.assert_frame_equal(result, expected)