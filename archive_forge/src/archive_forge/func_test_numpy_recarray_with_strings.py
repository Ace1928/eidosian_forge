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
def test_numpy_recarray_with_strings():
    x = np.array([('Alice', 1), ('Bob', 2)], dtype=[('name', 'O'), ('amt', 'i4')])
    assert discover(x) == dshape('2 * {name: string, amt: int32}')