from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import Linear, NumpyOps, Relu, chain
def get_predict(layer, param_name, inputs):
    """Helper for gradient check. To do the numeric gradient check, we have
    to be able to wiggle one value in a parameter, and check the prediction
    before and after. So we need to get a callback that gives an output
    given a change to one weight.
    """

    def predict(i, epsilon):
        param = layer.get_param(param_name)
        shape = param.shape
        param = param.ravel()
        param[i] += epsilon
        layer.set_param(param_name, param.reshape(shape))
        outputs = layer.predict(inputs)
        param[i] -= epsilon
        layer.set_param(param_name, param.reshape(shape))
        return outputs.reshape(shape)
    return predict