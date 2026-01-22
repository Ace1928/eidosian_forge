from __future__ import annotations
import csv
from io import (
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
def test_malformed_skipfooter(python_parser_only):
    parser = python_parser_only
    data = 'ignore\nA,B,C\n1,2,3 # comment\n1,2,3,4,5\n2,3,4\nfooter\n'
    msg = 'Expected 3 fields in line 4, saw 5'
    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(data), header=1, comment='#', skipfooter=1)