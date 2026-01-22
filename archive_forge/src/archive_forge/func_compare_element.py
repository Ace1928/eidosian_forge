from __future__ import annotations
from array import array
import bz2
import datetime
import functools
from functools import partial
import gzip
import io
import os
from pathlib import Path
import pickle
import shutil
import tarfile
from typing import Any
import uuid
import zipfile
import numpy as np
import pytest
from pandas.compat import (
from pandas.compat._optional import import_optional_dependency
from pandas.compat.compressors import flatten_buffer
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.generate_legacy_storage_files import create_pickle_data
import pandas.io.common as icom
from pandas.tseries.offsets import (
def compare_element(result, expected, typ):
    if isinstance(expected, Index):
        tm.assert_index_equal(expected, result)
        return
    if typ.startswith('sp_'):
        tm.assert_equal(result, expected)
    elif typ == 'timestamp':
        if expected is pd.NaT:
            assert result is pd.NaT
        else:
            assert result == expected
    else:
        comparator = getattr(tm, f'assert_{typ}_equal', tm.assert_almost_equal)
        comparator(result, expected)