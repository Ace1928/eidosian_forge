from __future__ import annotations
import csv
from io import (
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('quoting', [csv.QUOTE_MINIMAL, csv.QUOTE_NONE])
def test_multi_char_sep_quotes(python_parser_only, quoting):
    kwargs = {'sep': ',,'}
    parser = python_parser_only
    data = 'a,,b\n1,,a\n2,,"2,,b"'
    if quoting == csv.QUOTE_NONE:
        msg = 'Expected 2 fields in line 3, saw 3'
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), quoting=quoting, **kwargs)
    else:
        msg = 'ignored when a multi-char delimiter is used'
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), quoting=quoting, **kwargs)