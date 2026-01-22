import csv
import functools
import itertools
import math
import os
import re
from io import BytesIO
from pathlib import Path
from string import ascii_letters
from typing import Union
import numpy as np
import pandas
import psutil
import pytest
from pandas.core.dtypes.common import (
import modin.pandas as pd
from modin.config import (
from modin.pandas.io import to_pandas
from modin.pandas.testing import (
from modin.utils import try_cast_to_pandas
def modin_df_almost_equals_pandas(modin_df, pandas_df, max_diff=0.0001):
    df_categories_equals(modin_df._to_pandas(), pandas_df)
    modin_df = to_pandas(modin_df)
    if hasattr(modin_df, 'select_dtypes'):
        modin_df = modin_df.select_dtypes(exclude=['category'])
    if hasattr(pandas_df, 'select_dtypes'):
        pandas_df = pandas_df.select_dtypes(exclude=['category'])
    if modin_df.equals(pandas_df):
        return
    isna = modin_df.isna().all()
    if isinstance(isna, bool):
        if isna:
            assert pandas_df.isna().all()
            return
    elif isna.all():
        assert pandas_df.isna().all().all()
        return
    diff = (modin_df - pandas_df).abs()
    diff /= pandas_df.abs()
    diff_max = diff.max() if isinstance(diff, pandas.Series) else diff.max().max()
    assert diff_max < max_diff, f'{diff_max} >= {max_diff}'