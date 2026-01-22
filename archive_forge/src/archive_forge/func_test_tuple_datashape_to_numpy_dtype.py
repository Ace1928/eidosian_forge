from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_tuple_datashape_to_numpy_dtype():
    ds = dshape('5 * (int32, float32)')
    assert to_numpy_dtype(ds) == [('f0', 'i4'), ('f1', 'f4')]