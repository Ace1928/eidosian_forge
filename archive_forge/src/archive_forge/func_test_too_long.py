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
def test_too_long(self):
    with option_context('display.precision', 4):
        df = DataFrame({'x': [12345.6789]})
        assert str(df) == '            x\n0  12345.6789'
        df = DataFrame({'x': [2000000.0]})
        assert str(df) == '           x\n0  2000000.0'
        df = DataFrame({'x': [12345.6789, 2000000.0]})
        assert str(df) == '            x\n0  1.2346e+04\n1  2.0000e+06'