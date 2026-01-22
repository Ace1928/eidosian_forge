from io import StringIO
import os
from pathlib import Path
import pytest
from pandas.errors import ParserError
import pandas._testing as tm
from pandas.io.parsers import read_csv
import pandas.io.parsers.readers as parsers
def test_close_file_handle_on_invalid_usecols(all_parsers):
    parser = all_parsers
    error = ValueError
    if parser.engine == 'pyarrow':
        pytest.skip(reason='https://github.com/apache/arrow/issues/38676')
    with tm.ensure_clean('test.csv') as fname:
        Path(fname).write_text('col1,col2\na,b\n1,2', encoding='utf-8')
        with tm.assert_produces_warning(False):
            with pytest.raises(error, match='col3'):
                parser.read_csv(fname, usecols=['col1', 'col2', 'col3'])
        os.unlink(fname)