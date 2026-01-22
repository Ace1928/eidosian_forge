from collections import OrderedDict
from itertools import starmap
from types import MappingProxyType
from warnings import catch_warnings, simplefilter
import numpy as np
import pytest
from datashader.datashape.discovery import (
from datashader.datashape.coretypes import (
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape import dshape
from datetime import date, time, datetime, timedelta
def test_nested_complex_record_type():
    dt = np.dtype([('a', 'U7'), ('b', [('c', 'int64', 2), ('d', 'float64')])])
    x = np.zeros(5, dt)
    s = "5 * {a: string[7, 'U32'], b: {c: 2 * int64, d: float64}}"
    assert discover(x) == dshape(s)