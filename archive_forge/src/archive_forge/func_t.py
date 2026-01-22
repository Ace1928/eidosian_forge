import pytest
import random
import networkx as nx
from networkx.algorithms import approximation as approx
from networkx.algorithms import threshold
def t(f, *args, **kwds):
    """call one function and check if global RNG changed"""
    global progress
    progress += 1
    print(progress, ',', end='')
    f(*args, **kwds)
    after_np_rv = np.random.rand()
    assert np_rv == after_np_rv
    np.random.seed(42)
    after_py_rv = random.random()
    assert py_rv == after_py_rv
    random.seed(42)