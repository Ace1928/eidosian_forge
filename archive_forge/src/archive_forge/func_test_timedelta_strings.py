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
def test_timedelta_strings():
    inputs = ['1 day', '-2 hours', '3 seconds', '1 microsecond', '1003 milliseconds']
    for ts in inputs:
        assert discover(ts) == TimeDelta(unit=ts.split()[1])
    with pytest.raises(ValueError):
        TimeDelta(unit='buzz light-years')