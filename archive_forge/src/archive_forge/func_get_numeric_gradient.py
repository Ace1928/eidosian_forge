from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import Linear, NumpyOps, Relu, chain
def get_numeric_gradient(predict, n, target):
    gradient = numpy.zeros(n)
    for i in range(n):
        out1 = predict(i, 0.0001)
        out2 = predict(i, -0.0001)
        err1 = _get_loss(out1, target)
        err2 = _get_loss(out2, target)
        gradient[i] = (err1 - err2) / (2 * 0.0001)
        print('NGrad', i, err1, err2)
    return gradient