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
@pytest.mark.parametrize(('freq', 'expected'), [('M', MonthEnd()), ('2M', MonthEnd(n=2)), ('ME', MonthEnd()), ('2ME', MonthEnd(n=2)), ('MS', MonthBegin()), ('2MS', MonthBegin(n=2)), ('D', Day()), ('2D', Day(n=2)), ('H', Hour()), ('2H', Hour(n=2)), ('h', Hour()), ('2h', Hour(n=2)), ('T', Minute()), ('2T', Minute(n=2)), ('min', Minute()), ('2min', Minute(n=2)), ('S', Second()), ('2S', Second(n=2)), ('L', Millisecond(n=1)), ('2L', Millisecond(n=2)), ('ms', Millisecond(n=1)), ('2ms', Millisecond(n=2)), ('U', Microsecond(n=1)), ('2U', Microsecond(n=2)), ('us', Microsecond(n=1)), ('2us', Microsecond(n=2)), ('-2M', MonthEnd(n=-2)), ('-2ME', MonthEnd(n=-2)), ('-2MS', MonthBegin(n=-2)), ('-2D', Day(n=-2)), ('-2H', Hour(n=-2)), ('-2h', Hour(n=-2)), ('-2T', Minute(n=-2)), ('-2min', Minute(n=-2)), ('-2S', Second(n=-2)), ('-2L', Millisecond(n=-2)), ('-2ms', Millisecond(n=-2)), ('-2U', Microsecond(n=-2)), ('-2us', Microsecond(n=-2))], ids=_id_func)
@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_to_offset_sub_annual(freq, expected):
    assert to_offset(freq) == expected