import threading
from pytest import fixture, raises
import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context
def test_multi_skts_single_ctx():

    @context()
    @socket(zmq.PUB)
    @socket(zmq.SUB)
    @socket(zmq.PUSH)
    def test(ctx, pub, sub, push):
        assert isinstance(ctx, zmq.Context), ctx
        assert isinstance(pub, zmq.Socket), pub
        assert isinstance(sub, zmq.Socket), sub
        assert isinstance(push, zmq.Socket), push
        assert pub.context is ctx
        assert sub.context is ctx
        assert push.context is ctx
        assert pub.type == zmq.PUB
        assert sub.type == zmq.SUB
        assert push.type == zmq.PUSH
    test()