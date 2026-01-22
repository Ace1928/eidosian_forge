import copy
import gc
import sys
import time
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest, skip_pypy
def test_buffer_in(self):
    """test using a buffer as input"""
    ins = '§§¶•ªº˜µ¬˚…∆˙åß∂©œ∑´†≈ç√'.encode()
    zmq.Frame(memoryview(ins))