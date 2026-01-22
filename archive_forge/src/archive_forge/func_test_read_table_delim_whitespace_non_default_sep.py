from datetime import datetime
from inspect import signature
from io import StringIO
import os
from pathlib import Path
import sys
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
from pandas.io.parsers import TextFileReader
from pandas.io.parsers.c_parser_wrapper import CParserWrapper
@pytest.mark.parametrize('delimiter', [',', '\t'])
def test_read_table_delim_whitespace_non_default_sep(all_parsers, delimiter):
    f = StringIO('a  b  c\n1 -2 -3\n4  5   6')
    parser = all_parsers
    msg = 'Specified a delimiter with both sep and delim_whitespace=True; you can only specify one.'
    depr_msg = "The 'delim_whitespace' keyword in pd.read_table is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=depr_msg, check_stacklevel=False):
        with pytest.raises(ValueError, match=msg):
            parser.read_table(f, delim_whitespace=True, sep=delimiter)
        with pytest.raises(ValueError, match=msg):
            parser.read_table(f, delim_whitespace=True, delimiter=delimiter)