import csv
from io import StringIO
import pytest
from pandas.compat import PY311
from pandas.errors import ParserError
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('kwargs,exp_data', [({}, [[1, 2, 'foo']]), ({'quotechar': '"', 'quoting': csv.QUOTE_MINIMAL}, [[1, 2, 'foo']]), ({'quotechar': '"', 'quoting': csv.QUOTE_ALL}, [[1, 2, 'foo']]), ({'quotechar': '"', 'quoting': csv.QUOTE_NONE}, [[1, 2, '"foo"']]), ({'quotechar': '"', 'quoting': csv.QUOTE_NONNUMERIC}, [[1.0, 2.0, 'foo']])])
@xfail_pyarrow
def test_quoting_various(all_parsers, kwargs, exp_data):
    data = '1,2,"foo"'
    parser = all_parsers
    columns = ['a', 'b', 'c']
    result = parser.read_csv(StringIO(data), names=columns, **kwargs)
    expected = DataFrame(exp_data, columns=columns)
    tm.assert_frame_equal(result, expected)