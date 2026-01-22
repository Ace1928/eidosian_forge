from __future__ import annotations
from itertools import product
from typing import Callable, Literal
import numpy as np
import pandas as pd
import pytest
from xarray import CFTimeIndex
from xarray.coding.cftime_offsets import (
from xarray.coding.frequencies import infer_freq
from xarray.core.dataarray import DataArray
from xarray.tests import (
@pytest.mark.filterwarnings("ignore:'.' is deprecated ")
@pytest.mark.skipif(has_pandas_ge_2_2, reason='only for pandas lt 2.2')
@pytest.mark.parametrize('freq, expected', (('T', 'T'), ('min', 'min'), ('S', 'S'), ('s', 's'), ('L', 'L'), ('ms', 'ms'), ('U', 'U'), ('us', 'us'), ('bh', 'bh'), ('cbh', 'cbh'), ('ns', 'ns')))
def test_new_to_legacy_freq_pd_freq_passthrough(freq, expected):
    result = _new_to_legacy_freq(freq)
    assert result == expected