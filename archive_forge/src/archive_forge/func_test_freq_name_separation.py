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
def test_freq_name_separation(self):
    s = Series(np.random.default_rng(2).standard_normal(10), index=date_range('1/1/2000', periods=10), name=0)
    result = repr(s)
    assert 'Freq: D, Name: 0' in result