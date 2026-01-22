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
def test_output_display_precision_trailing_zeroes(self):
    with option_context('display.precision', 0):
        s = Series([840.0, 4200.0])
        expected_output = '0     840\n1    4200\ndtype: float64'
        assert str(s) == expected_output