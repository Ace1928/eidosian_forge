from statsmodels.compat.python import lrange
from io import BytesIO
from itertools import product
import numpy as np
from numpy.testing import assert_, assert_raises
import pandas as pd
import pytest
from statsmodels.api import datasets
from statsmodels.graphics.mosaicplot import (
def test_false_split():
    pure_square = [0.0, 0.0, 1.0, 1.0]
    conf_h = dict(proportion=[1], gap=0.0, horizontal=True)
    conf_v = dict(proportion=[1], gap=0.0, horizontal=False)
    eq(_split_rect(*pure_square, **conf_h), pure_square)
    eq(_split_rect(*pure_square, **conf_v), pure_square)
    conf_h = dict(proportion=[1], gap=0.5, horizontal=True)
    conf_v = dict(proportion=[1], gap=0.5, horizontal=False)
    eq(_split_rect(*pure_square, **conf_h), pure_square)
    eq(_split_rect(*pure_square, **conf_v), pure_square)
    null_square = [0.0, 0.0, 0.0, 0.0]
    conf = dict(proportion=[1], gap=0.0, horizontal=True)
    eq(_split_rect(*null_square, **conf), null_square)
    conf = dict(proportion=[1], gap=1.0, horizontal=True)
    eq(_split_rect(*null_square, **conf), null_square)
    neg_square = [0.0, 0.0, -1.0, 0.0]
    conf = dict(proportion=[1], gap=0.0, horizontal=True)
    assert_raises(ValueError, _split_rect, *neg_square, **conf)
    conf = dict(proportion=[1, 1], gap=0.0, horizontal=True)
    assert_raises(ValueError, _split_rect, *neg_square, **conf)
    conf = dict(proportion=[1], gap=0.5, horizontal=True)
    assert_raises(ValueError, _split_rect, *neg_square, **conf)
    conf = dict(proportion=[1, 1], gap=0.5, horizontal=True)
    assert_raises(ValueError, _split_rect, *neg_square, **conf)