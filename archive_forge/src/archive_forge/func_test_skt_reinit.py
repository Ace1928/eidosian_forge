import threading
from pytest import fixture, raises
import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context
def test_skt_reinit():
    result = {'foo': None, 'bar': None}

    @socket(zmq.PUB)
    def f(key, skt):
        assert isinstance(skt, zmq.Socket), skt
        result[key] = skt
    foo_t = threading.Thread(target=f, args=('foo',))
    bar_t = threading.Thread(target=f, args=('bar',))
    foo_t.start()
    bar_t.start()
    foo_t.join()
    bar_t.join()
    assert result['foo'] is not None, result
    assert result['bar'] is not None, result
    assert result['foo'] is not result['bar'], result