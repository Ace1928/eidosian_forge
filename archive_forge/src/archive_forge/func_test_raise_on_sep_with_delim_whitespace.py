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
def test_raise_on_sep_with_delim_whitespace(all_parsers):
    data = 'a b c\n1 2 3'
    parser = all_parsers
    depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"
    with pytest.raises(ValueError, match='you can only specify one'):
        with tm.assert_produces_warning(FutureWarning, match=depr_msg, check_stacklevel=False):
            parser.read_csv(StringIO(data), sep='\\s', delim_whitespace=True)