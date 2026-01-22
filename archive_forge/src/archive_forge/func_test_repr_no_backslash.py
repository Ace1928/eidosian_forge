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
def test_repr_no_backslash(self):
    with option_context('mode.sim_interactive', True):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        assert '\\' not in repr(df)