import csv
from io import StringIO
import pytest
from pandas.compat import PY311
from pandas.errors import ParserError
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('kwargs,msg', [({'quotechar': 'foo'}, '"quotechar" must be a(n)? 1-character string'), ({'quotechar': None, 'quoting': csv.QUOTE_MINIMAL}, 'quotechar must be set if quoting enabled'), ({'quotechar': 2}, '"quotechar" must be string( or None)?, not int')])
@skip_pyarrow
def test_bad_quote_char(all_parsers, kwargs, msg):
    data = '1,2,3'
    parser = all_parsers
    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), **kwargs)