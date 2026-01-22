from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import Linear, NumpyOps, Relu, chain
def get_expected_predict(input_data, Ws, bs):
    numpy_ops = NumpyOps()
    X = input_data
    for i, (W, b) in enumerate(zip(Ws, bs)):
        X = numpy_ops.asarray(X)
        if i > 0:
            X *= X > 0
        X = numpy.tensordot(X, W, axes=[[1], [1]]) + b
    return X