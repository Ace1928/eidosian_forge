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
def test_Microsecond_multiplied_float_error():
    """Test that the appropriate error is raised if a Tick offset is multiplied
    by a float which causes it not to be representable by a
    microsecond-precision timedelta."""
    with pytest.raises(ValueError, match='Could not convert to integer offset at any resolution'):
        Microsecond() * 0.5