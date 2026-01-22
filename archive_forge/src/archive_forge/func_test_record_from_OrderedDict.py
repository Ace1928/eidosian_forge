from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_record_from_OrderedDict():
    r = Record(OrderedDict([('a', 'int32'), ('b', 'float64')]))
    assert r.to_numpy_dtype() == np.dtype([('a', 'i4'), ('b', 'f8')])