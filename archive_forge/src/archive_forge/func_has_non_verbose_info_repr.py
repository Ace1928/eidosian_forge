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
def has_non_verbose_info_repr(df):
    has_info = has_info_repr(df)
    r = repr(df)
    nv = len(r.split('\n')) == 6
    return has_info and nv