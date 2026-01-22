import copy
import gc
import sys
import time
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest, skip_pypy
def test_buffer_numpy(self):
    """test non-copying numpy array messages"""
    try:
        import numpy
        from numpy.testing import assert_array_equal
    except ImportError:
        raise SkipTest('requires numpy')
    rand = numpy.random.randint
    shapes = [rand(2, 5) for i in range(5)]
    a, b = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
    dtypes = [int, float, '>i4', 'B']
    for i in range(1, len(shapes) + 1):
        shape = shapes[:i]
        for dt in dtypes:
            A = numpy.empty(shape, dtype=dt)
            a.send(A, copy=False)
            msg = b.recv(copy=False)
            B = numpy.frombuffer(msg, A.dtype).reshape(A.shape)
            assert_array_equal(A, B)
        A = numpy.empty(shape, dtype=[('a', int), ('b', float), ('c', 'a32')])
        A['a'] = 1024
        A['b'] = 1000000000.0
        A['c'] = 'hello there'
        a.send(A, copy=False)
        msg = b.recv(copy=False)
        B = numpy.frombuffer(msg, A.dtype).reshape(A.shape)
        assert_array_equal(A, B)