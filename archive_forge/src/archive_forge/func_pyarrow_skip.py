from __future__ import annotations
import os
import pytest
from pandas.compat._optional import VERSIONS
from pandas import (
import pandas._testing as tm
@pytest.fixture
def pyarrow_skip(request):
    """
    Fixture that skips a test if the engine is pyarrow.

    Use if failure is do a parsing failure from pyarrow.csv.read_csv
    """
    if 'all_parsers' in request.fixturenames:
        parser = request.getfixturevalue('all_parsers')
    elif 'all_parsers_all_precisions' in request.fixturenames:
        parser = request.getfixturevalue('all_parsers_all_precisions')[0]
    else:
        return
    if parser.engine == 'pyarrow':
        pytest.skip(reason='https://github.com/apache/arrow/issues/38676')