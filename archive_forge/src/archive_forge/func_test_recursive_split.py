from statsmodels.compat.python import lrange
from io import BytesIO
from itertools import product
import numpy as np
from numpy.testing import assert_, assert_raises
import pandas as pd
import pytest
from statsmodels.api import datasets
from statsmodels.graphics.mosaicplot import (
def test_recursive_split():
    keys = list(product('mf'))
    data = dict(zip(keys, [1] * len(keys)))
    res = _hierarchical_split(data, gap=0)
    assert_(list(res.keys()) == keys)
    res['m',] = (0.0, 0.0, 0.5, 1.0)
    res['f',] = (0.5, 0.0, 0.5, 1.0)
    keys = list(product('mf', 'yao'))
    data = dict(zip(keys, [1] * len(keys)))
    res = _hierarchical_split(data, gap=0)
    assert_(list(res.keys()) == keys)
    res['m', 'y'] = (0.0, 0.0, 0.5, 1 / 3)
    res['m', 'a'] = (0.0, 1 / 3, 0.5, 1 / 3)
    res['m', 'o'] = (0.0, 2 / 3, 0.5, 1 / 3)
    res['f', 'y'] = (0.5, 0.0, 0.5, 1 / 3)
    res['f', 'a'] = (0.5, 1 / 3, 0.5, 1 / 3)
    res['f', 'o'] = (0.5, 2 / 3, 0.5, 1 / 3)