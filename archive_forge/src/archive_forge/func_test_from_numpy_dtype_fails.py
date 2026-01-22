from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
@pytest.mark.xfail(raises=NotImplementedError, reason='DataShape does not know about void types (yet?)')
def test_from_numpy_dtype_fails():
    x = np.zeros(2, np.dtype([('a', 'int32')]))
    CType.from_numpy_dtype(x[0].dtype)