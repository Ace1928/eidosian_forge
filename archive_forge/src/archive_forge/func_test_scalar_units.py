from __future__ import annotations
import contextlib
import warnings
import numpy as np
import pandas as pd
import pytest
from xarray import (
from xarray.backends.common import WritableCFDataStore
from xarray.backends.memory import InMemoryDataStore
from xarray.conventions import decode_cf
from xarray.testing import assert_identical
from xarray.tests import (
from xarray.tests.test_backends import CFEncodedBase
def test_scalar_units() -> None:
    var = Variable(['t'], [np.nan, np.nan, 2], {'units': np.nan})
    actual = conventions.decode_cf_variable('t', var)
    assert_identical(actual, var)