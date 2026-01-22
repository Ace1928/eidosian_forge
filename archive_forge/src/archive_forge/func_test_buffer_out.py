import copy
import gc
import sys
import time
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest, skip_pypy
def test_buffer_out(self):
    """receiving buffered output"""
    ins = '§§¶•ªº˜µ¬˚…∆˙åß∂©œ∑´†≈ç√'.encode()
    m = zmq.Frame(ins)
    outb = m.buffer
    assert isinstance(outb, memoryview)
    assert outb is m.buffer
    assert m.buffer is m.buffer