import numpy
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, just, tuples
from thinc.api import Linear, NumpyOps
def ndarrays(min_len=0, max_len=10, min_val=-10.0, max_val=10.0):
    return lengths(lo=1, hi=2).flatmap(lambda n: ndarrays_of_shape(n, lo=min_val, hi=max_val))