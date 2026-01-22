from statsmodels.compat.python import lrange
from io import BytesIO
from itertools import product
import numpy as np
from numpy.testing import assert_, assert_raises
import pandas as pd
import pytest
from statsmodels.api import datasets
from statsmodels.graphics.mosaicplot import (
def test__key_splitting():
    base_rect = {tuple(): (0, 0, 1, 1)}
    res = _key_splitting(base_rect, ['a', 'b'], [1, 1], tuple(), True, 0)
    assert_(list(res.keys()) == [('a',), ('b',)])
    eq(res['a',], (0, 0, 0.5, 1))
    eq(res['b',], (0.5, 0, 0.5, 1))
    res_bis = _key_splitting(res, ['c', 'd'], [1, 1], ('a',), False, 0)
    assert_(list(res_bis.keys()) == [('a', 'c'), ('a', 'd'), ('b',)])
    eq(res_bis['a', 'c'], (0.0, 0.0, 0.5, 0.5))
    eq(res_bis['a', 'd'], (0.0, 0.5, 0.5, 0.5))
    eq(res_bis['b',], (0.5, 0, 0.5, 1))
    base_rect = {('total',): (0, 0, 1, 1)}
    res = _key_splitting(base_rect, ['a', 'b'], [1, 2], ('total',), True, 0)
    assert_(list(res.keys()) == [('total',) + (e,) for e in ['a', 'b']])
    eq(res['total', 'a'], (0, 0, 1 / 3, 1))
    eq(res['total', 'b'], (1 / 3, 0, 2 / 3, 1))