import threading
from pytest import fixture, raises
import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context
def test_ctx_skt():

    @context()
    @socket(zmq.PUB)
    def test(ctx, skt):
        assert isinstance(ctx, zmq.Context), ctx
        assert isinstance(skt, zmq.Socket), skt
        assert skt.type == zmq.PUB
    test()