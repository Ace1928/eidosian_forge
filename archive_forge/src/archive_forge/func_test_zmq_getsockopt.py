import time
from unittest import TestCase
from zmq.tests import SkipTest
def test_zmq_getsockopt(self):
    ctx = C.zmq_ctx_new()
    socket = C.zmq_socket(ctx, PUSH)
    identity = ffi.new('char[]', b'zmq')
    ret = C.zmq_setsockopt(socket, IDENTITY, ffi.cast('void*', identity), 3)
    assert ret == 0
    option_len = ffi.new('size_t*', 3)
    option = ffi.new('char[3]')
    ret = C.zmq_getsockopt(socket, IDENTITY, ffi.cast('void*', option), option_len)
    assert ret == 0
    assert ffi.string(ffi.cast('char*', option))[0:1] == b'z'
    assert ffi.string(ffi.cast('char*', option))[1:2] == b'm'
    assert ffi.string(ffi.cast('char*', option))[2:3] == b'q'
    assert ctx != ffi.NULL
    assert ffi.NULL != socket
    assert 0 == C.zmq_close(socket)
    assert 0 == C.zmq_ctx_destroy(ctx)