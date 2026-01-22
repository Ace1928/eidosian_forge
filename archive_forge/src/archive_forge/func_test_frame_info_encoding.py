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
def test_frame_info_encoding(self):
    index = ["'Til There Was You (1997)", 'ldum klaka (Cold Fever) (1994)']
    with option_context('display.max_rows', 1):
        df = DataFrame(columns=['a', 'b', 'c'], index=index)
        repr(df)
        repr(df.T)