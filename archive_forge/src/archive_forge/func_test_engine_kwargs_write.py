import contextlib
from pathlib import Path
import re
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.excel import (
from pandas.io.excel._openpyxl import OpenpyxlReader
@pytest.mark.parametrize('iso_dates', [True, False])
def test_engine_kwargs_write(ext, iso_dates):
    engine_kwargs = {'iso_dates': iso_dates}
    with tm.ensure_clean(ext) as f:
        with ExcelWriter(f, engine='openpyxl', engine_kwargs=engine_kwargs) as writer:
            assert writer.book.iso_dates == iso_dates
            DataFrame().to_excel(writer)