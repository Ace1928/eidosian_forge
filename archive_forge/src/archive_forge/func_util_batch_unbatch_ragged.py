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
def util_batch_unbatch_ragged(model: Model[Ragged, Array2d], in_data: Ragged, out_data: Array2d):
    with data_validation(True):
        model.initialize(in_data, out_data)
        Y_batched = model.predict(in_data)
        Y_not_batched = [model.predict(in_data[i])[0] for i in range(len(in_data))]
        assert_almost_equal(Y_batched, Y_not_batched, decimal=4)