from typing import List, Optional
import numpy
import pytest
import srsly
from numpy.testing import assert_almost_equal
from thinc.api import Dropout, Model, NumpyOps, registry, with_padded
from thinc.backends import NumpyOps
from thinc.compat import has_torch
from thinc.types import Array2d, Floats2d, FloatsXd, Padded, Ragged, Shape
from thinc.util import data_validation, get_width
def assert_data_match(Y, out_data):
    assert type(Y) == type(out_data)
    if isinstance(out_data, OPS.xp.ndarray):
        assert isinstance(Y, OPS.xp.ndarray)
        assert out_data.ndim == Y.ndim
    elif isinstance(out_data, Ragged):
        assert isinstance(Y, Ragged)
        assert out_data.data.ndim == Y.data.ndim
        assert out_data.lengths.ndim == Y.lengths.ndim
    elif isinstance(out_data, Padded):
        assert isinstance(Y, Padded)
        assert out_data.data.ndim == Y.data.ndim
        assert out_data.size_at_t.ndim == Y.size_at_t.ndim
        assert len(out_data.lengths) == len(Y.lengths)
        assert len(out_data.indices) == len(Y.indices)
    elif isinstance(out_data, (list, tuple)):
        assert isinstance(Y, (list, tuple))
        assert all((isinstance(x, numpy.ndarray) for x in Y))
    else:
        pytest.fail(f'wrong output of {type(Y)}: {Y}')