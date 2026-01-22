from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import Linear, NumpyOps, Relu, chain
def numeric_gradient(predict, weights, epsilon=0.0001):
    out1 = predict(weights + epsilon)
    out2 = predict(weights - epsilon)
    return (out1 - out2) / (2 * epsilon)