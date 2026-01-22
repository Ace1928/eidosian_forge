import csv
from io import StringIO
import pytest
from pandas.errors import ParserWarning
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('arg', [None, 'doublequote', 'escapechar', 'skipinitialspace', 'quotechar', 'quoting'])
@pytest.mark.parametrize('value', ['dialect', 'default', 'other'])
def test_dialect_conflict_except_delimiter(all_parsers, custom_dialect, arg, value):
    dialect_name, dialect_kwargs = custom_dialect
    parser = all_parsers
    expected = DataFrame({'a': [1], 'b': [2]})
    data = 'a:b\n1:2'
    warning_klass = None
    kwds = {}
    if arg is not None:
        if value == 'dialect':
            kwds[arg] = dialect_kwargs[arg]
        elif value == 'default':
            from pandas.io.parsers.base_parser import parser_defaults
            kwds[arg] = parser_defaults[arg]
        else:
            warning_klass = ParserWarning
            kwds[arg] = 'blah'
    with tm.with_csv_dialect(dialect_name, **dialect_kwargs):
        if parser.engine == 'pyarrow':
            msg = "The 'dialect' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv_check_warnings(None, 'Conflicting values for', StringIO(data), dialect=dialect_name, **kwds)
            return
        result = parser.read_csv_check_warnings(warning_klass, 'Conflicting values for', StringIO(data), dialect=dialect_name, **kwds)
        tm.assert_frame_equal(result, expected)