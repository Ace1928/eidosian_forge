import time
from unittest import TestCase
from zmq.tests import SkipTest
def test_zmq_poll(self):
    ctx = C.zmq_ctx_new()
    sender = C.zmq_socket(ctx, REQ)
    receiver = C.zmq_socket(ctx, REP)
    r1 = C.zmq_bind(receiver, b'tcp://*:3333')
    r2 = C.zmq_connect(sender, b'tcp://127.0.0.1:3333')
    zmq_msg = ffi.new('zmq_msg_t*')
    message = ffi.new('char[5]', b'Hello')
    C.zmq_msg_init_data(zmq_msg, ffi.cast('void*', message), ffi.cast('size_t', 5), ffi.NULL, ffi.NULL)
    receiver_pollitem = ffi.new('zmq_pollitem_t*')
    receiver_pollitem.socket = receiver
    receiver_pollitem.fd = 0
    receiver_pollitem.events = POLLIN | POLLOUT
    receiver_pollitem.revents = 0
    ret = C.zmq_poll(ffi.NULL, 0, 0)
    assert ret == 0
    ret = C.zmq_poll(receiver_pollitem, 1, 0)
    assert ret == 0
    ret = C.zmq_msg_send(zmq_msg, sender, 0)
    print(ffi.string(C.zmq_strerror(C.zmq_errno())))
    assert ret == 5
    time.sleep(0.2)
    ret = C.zmq_poll(receiver_pollitem, 1, 0)
    assert ret == 1
    assert int(receiver_pollitem.revents) & POLLIN
    assert not int(receiver_pollitem.revents) & POLLOUT
    zmq_msg2 = ffi.new('zmq_msg_t*')
    C.zmq_msg_init(zmq_msg2)
    ret_recv = C.zmq_msg_recv(zmq_msg2, receiver, 0)
    assert ret_recv == 5
    assert 5 == C.zmq_msg_size(zmq_msg2)
    assert b'Hello' == ffi.buffer(C.zmq_msg_data(zmq_msg2), C.zmq_msg_size(zmq_msg2))[:]
    sender_pollitem = ffi.new('zmq_pollitem_t*')
    sender_pollitem.socket = sender
    sender_pollitem.fd = 0
    sender_pollitem.events = POLLIN | POLLOUT
    sender_pollitem.revents = 0
    ret = C.zmq_poll(sender_pollitem, 1, 0)
    assert ret == 0
    zmq_msg_again = ffi.new('zmq_msg_t*')
    message_again = ffi.new('char[11]', b'Hello Again')
    C.zmq_msg_init_data(zmq_msg_again, ffi.cast('void*', message_again), ffi.cast('size_t', 11), ffi.NULL, ffi.NULL)
    assert 11 == C.zmq_msg_send(zmq_msg_again, receiver, 0)
    time.sleep(0.2)
    assert 0 <= C.zmq_poll(sender_pollitem, 1, 0)
    assert int(sender_pollitem.revents) & POLLIN
    assert 11 == C.zmq_msg_recv(zmq_msg2, sender, 0)
    assert 11 == C.zmq_msg_size(zmq_msg2)
    assert b'Hello Again' == ffi.buffer(C.zmq_msg_data(zmq_msg2), int(C.zmq_msg_size(zmq_msg2)))[:]
    assert 0 == C.zmq_close(sender)
    assert 0 == C.zmq_close(receiver)
    assert 0 == C.zmq_ctx_destroy(ctx)
    assert 0 == C.zmq_msg_close(zmq_msg)
    assert 0 == C.zmq_msg_close(zmq_msg2)
    assert 0 == C.zmq_msg_close(zmq_msg_again)