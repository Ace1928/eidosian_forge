import threading
from pytest import fixture, raises
import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context
def test_skt_multi_thread():

    @socket(zmq.PUB)
    @socket(zmq.SUB)
    @socket(zmq.PUSH)
    def f(pub, sub, push):
        assert isinstance(pub, zmq.Socket), pub
        assert isinstance(sub, zmq.Socket), sub
        assert isinstance(push, zmq.Socket), push
        assert pub.context is zmq.Context.instance()
        assert sub.context is zmq.Context.instance()
        assert push.context is zmq.Context.instance()
        assert pub.type == zmq.PUB
        assert sub.type == zmq.SUB
        assert push.type == zmq.PUSH
        assert len(set(map(id, [pub, sub, push]))) == 3
    threads = [threading.Thread(target=f) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]