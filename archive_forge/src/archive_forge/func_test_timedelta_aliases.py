from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_timedelta_aliases():
    for alias in _unit_aliases:
        a = alias + 's'
        assert dshape('timedelta[unit=%r]' % a) == dshape('timedelta[unit=%r]' % _unit_aliases[alias])