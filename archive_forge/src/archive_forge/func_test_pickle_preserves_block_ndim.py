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
@td.skip_array_manager_invalid_test
def test_pickle_preserves_block_ndim():
    ser = Series(list('abc')).astype('category').iloc[[0]]
    res = tm.round_trip_pickle(ser)
    assert res._mgr.blocks[0].ndim == 1
    assert res._mgr.blocks[0].shape == (1,)
    tm.assert_series_equal(res[[True]], ser)