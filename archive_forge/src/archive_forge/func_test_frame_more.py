import copy
import gc
import sys
import time
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest, skip_pypy
@skip_pypy
def test_frame_more(self):
    """test Frame.more attribute"""
    frame = zmq.Frame(b'hello')
    assert not frame.more
    sa, sb = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
    sa.send_multipart([b'hi', b'there'])
    frame = self.recv(sb, copy=False)
    assert frame.more
    if zmq.zmq_version_info()[0] >= 3 and (not PYPY):
        assert frame.get(zmq.MORE)
    frame = self.recv(sb, copy=False)
    assert not frame.more
    if zmq.zmq_version_info()[0] >= 3 and (not PYPY):
        assert not frame.get(zmq.MORE)