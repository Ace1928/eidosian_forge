import codecs
import csv
from io import StringIO
import os
from pathlib import Path
import numpy as np
import pytest
from pandas.compat import PY311
from pandas.errors import (
from pandas import DataFrame
import pandas._testing as tm
def test_invalid_on_bad_line(all_parsers):
    parser = all_parsers
    data = 'a\n1\n1,2,3\n4\n5,6,7'
    with pytest.raises(ValueError, match='Argument abc is invalid for on_bad_lines'):
        parser.read_csv(StringIO(data), on_bad_lines='abc')