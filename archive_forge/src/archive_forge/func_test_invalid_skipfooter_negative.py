from __future__ import annotations
import csv
from io import (
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
def test_invalid_skipfooter_negative(python_parser_only):
    data = 'a\n1\n2'
    parser = python_parser_only
    msg = 'skipfooter cannot be negative'
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), skipfooter=-1)