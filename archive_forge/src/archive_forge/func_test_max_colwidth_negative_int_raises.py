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
def test_max_colwidth_negative_int_raises(self):
    with pytest.raises(ValueError, match='Value must be a nonnegative integer or None'):
        with option_context('display.max_colwidth', -1):
            pass