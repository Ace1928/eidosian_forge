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
@pytest.mark.parametrize('name', [777, 777.0, 'name', datetime.datetime(2001, 11, 11), (1, 2)])
def test_pickle_preserve_name(name):
    unpickled = tm.round_trip_pickle(Series(np.arange(10, dtype=np.float64), name=name))
    assert unpickled.name == name