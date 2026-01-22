from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
@pytest.mark.parametrize(['base', 'expected'], zip([timedelta_, date_, datetime_], ['timedelta64[us]', 'datetime64[D]', 'datetime64[us]']))
def test_option_to_numpy_dtype(base, expected):
    assert Option(base).to_numpy_dtype() == np.dtype(expected)