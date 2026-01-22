import numpy
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, just, tuples
from thinc.api import Linear, NumpyOps
def ndarrays_of_shape(shape, lo=-10.0, hi=10.0, dtype='float32', width=32):
    if dtype.startswith('float'):
        return arrays(dtype, shape=shape, elements=floats(min_value=lo, max_value=hi, width=width))
    else:
        return arrays(dtype, shape=shape, elements=integers(min_value=lo, max_value=hi))