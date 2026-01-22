import copy
import gc
import sys
import time
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest, skip_pypy
def test_multi_tracker(self):
    m = zmq.Frame(b'asdf', copy=False, track=True)
    m2 = zmq.Frame(b'whoda', copy=False, track=True)
    mt = zmq.MessageTracker(m, m2)
    assert not m.tracker.done
    assert not mt.done
    self.assertRaises(zmq.NotDone, mt.wait, 0.1)
    del m
    for i in range(3):
        gc.collect()
    self.assertRaises(zmq.NotDone, mt.wait, 0.1)
    assert not mt.done
    del m2
    for i in range(3):
        gc.collect()
    assert mt.wait(0.1) is None
    assert mt.done