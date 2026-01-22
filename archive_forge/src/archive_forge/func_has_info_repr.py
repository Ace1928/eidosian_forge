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
def has_info_repr(df):
    r = repr(df)
    c1 = r.split('\n')[0].startswith('<class')
    c2 = r.split('\n')[0].startswith('&lt;class')
    return c1 or c2