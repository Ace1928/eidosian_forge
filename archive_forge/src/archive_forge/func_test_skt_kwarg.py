import threading
from pytest import fixture, raises
import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context
def test_skt_kwarg():

    @context()
    @socket(zmq.PUB, name='myskt')
    def test(ctx, myskt):
        assert isinstance(myskt, zmq.Socket), myskt
        assert isinstance(ctx, zmq.Context), ctx
        assert myskt.type == zmq.PUB
    test()