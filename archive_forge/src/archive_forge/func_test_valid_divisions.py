from __future__ import annotations
import re
import warnings
from collections.abc import Iterable
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import dask
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_300, tm
from dask.dataframe.core import apply_and_enforce
from dask.dataframe.utils import (
from dask.local import get_sync
@pytest.mark.parametrize('divisions, valid', [([1, 2, 3], True), ([3, 2, 1], False), ([1, 1, 1], False), ([0, 1, 1], True), ((1, 2, 3), True), (123, False), ([0, float('nan'), 1], False)])
def test_valid_divisions(divisions, valid):
    assert valid_divisions(divisions) == valid