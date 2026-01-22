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
@pytest.mark.parametrize('nrows', range(1, 6))
def test_blank_lines_between_header_and_data_rows(all_parsers, nrows):
    ref = DataFrame([[np.nan, np.nan], [np.nan, np.nan], [1, 2], [np.nan, np.nan], [3, 4]], columns=list('ab'))
    csv = '\nheader\n\na,b\n\n\n1,2\n\n3,4'
    parser = all_parsers
    if parser.engine == 'pyarrow':
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(csv), header=3, nrows=nrows, skip_blank_lines=False)
        return
    df = parser.read_csv(StringIO(csv), header=3, nrows=nrows, skip_blank_lines=False)
    tm.assert_frame_equal(df, ref[:nrows])