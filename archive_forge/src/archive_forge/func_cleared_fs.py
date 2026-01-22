import io
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
@pytest.fixture
def cleared_fs():
    fsspec = pytest.importorskip('fsspec')
    memfs = fsspec.filesystem('memory')
    yield memfs
    memfs.store.clear()