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
def test_read_csv_names_not_accepting_sets(all_parsers):
    data = '    1,2,3\n    4,5,6\n'
    parser = all_parsers
    with pytest.raises(ValueError, match='Names should be an ordered collection.'):
        parser.read_csv(StringIO(data), names=set('QAZ'))