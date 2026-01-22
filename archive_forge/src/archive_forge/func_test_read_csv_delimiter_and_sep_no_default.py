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
def test_read_csv_delimiter_and_sep_no_default(all_parsers):
    f = StringIO('a,b\n1,2')
    parser = all_parsers
    msg = 'Specified a sep and a delimiter; you can only specify one.'
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(f, sep=' ', delimiter='.')