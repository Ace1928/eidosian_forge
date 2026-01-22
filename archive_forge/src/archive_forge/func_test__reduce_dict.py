from statsmodels.compat.python import lrange
from io import BytesIO
from itertools import product
import numpy as np
from numpy.testing import assert_, assert_raises
import pandas as pd
import pytest
from statsmodels.api import datasets
from statsmodels.graphics.mosaicplot import (
def test__reduce_dict():
    data = dict(zip(list(product('mf', 'oy', 'wn')), [1] * 8))
    eq(_reduce_dict(data, ('m',)), 4)
    eq(_reduce_dict(data, ('m', 'o')), 2)
    eq(_reduce_dict(data, ('m', 'o', 'w')), 1)
    data = dict(zip(list(product('mf', 'oy', 'wn')), lrange(8)))
    eq(_reduce_dict(data, ('m',)), 6)
    eq(_reduce_dict(data, ('m', 'o')), 1)
    eq(_reduce_dict(data, ('m', 'o', 'w')), 0)