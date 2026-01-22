from statsmodels.compat.python import lrange
from io import BytesIO
from itertools import product
import numpy as np
from numpy.testing import assert_, assert_raises
import pandas as pd
import pytest
from statsmodels.api import datasets
from statsmodels.graphics.mosaicplot import (
def test_proportion_normalization():
    eq(_normalize_split(0.0), [0.0, 0.0, 1.0])
    eq(_normalize_split(1.0), [0.0, 1.0, 1.0])
    eq(_normalize_split(2.0), [0.0, 1.0, 1.0])
    assert_raises(ValueError, _normalize_split, -1)
    assert_raises(ValueError, _normalize_split, [1.0, -1])
    assert_raises(ValueError, _normalize_split, [1.0, -1, 0.0])
    assert_raises(ValueError, _normalize_split, [0.0])
    assert_raises(ValueError, _normalize_split, [0.0, 0.0])
    eq(_normalize_split([0.5]), [0.0, 1.0])
    eq(_normalize_split([1.0]), [0.0, 1.0])
    eq(_normalize_split([2.0]), [0.0, 1.0])
    for x in [0.3, 0.5, 0.9]:
        eq(_normalize_split(x), [0.0, x, 1.0])
    for x, y in [(0.25, 0.5), (0.1, 0.8), (10.0, 30.0)]:
        eq(_normalize_split([x, y]), [0.0, x / (x + y), 1.0])
    for x, y, z in [(1.0, 1.0, 1.0), (0.1, 0.5, 0.7), (10.0, 30.0, 40)]:
        eq(_normalize_split([x, y, z]), [0.0, x / (x + y + z), (x + y) / (x + y + z), 1.0])