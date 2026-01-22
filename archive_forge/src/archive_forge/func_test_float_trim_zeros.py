from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
def test_float_trim_zeros(self):
    vals = [20843091730.5, 35220501730.5, 23067481730.5, 20395421730.5, 55989781730.5]
    for line in repr(Series(vals)).split('\n'):
        if line.startswith('dtype:'):
            continue
        if _three_digit_exp():
            assert '+010' in line
        else:
            assert '+10' in line