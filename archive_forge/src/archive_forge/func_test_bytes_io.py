from datetime import (
from functools import partial
from io import BytesIO
import os
import re
import numpy as np
import pytest
from pandas.compat import is_platform_windows
from pandas.compat._constants import PY310
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.excel import (
from pandas.io.excel._util import _writers
def test_bytes_io(self, engine):
    with BytesIO() as bio:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        with ExcelWriter(bio, engine=engine) as writer:
            df.to_excel(writer)
        bio.seek(0)
        reread_df = pd.read_excel(bio, index_col=0)
        tm.assert_frame_equal(df, reread_df)