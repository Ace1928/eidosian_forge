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
def test_string_repr_encoding(self, datapath):
    filepath = datapath('io', 'parser', 'data', 'unicode_series.csv')
    df = read_csv(filepath, header=None, encoding='latin1')
    repr(df)
    repr(df[1])