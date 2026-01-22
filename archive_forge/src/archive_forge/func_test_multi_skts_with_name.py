import threading
from pytest import fixture, raises
import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context
def test_multi_skts_with_name():

    @socket('foo', zmq.PUSH)
    @socket('bar', zmq.SUB)
    @socket('baz', zmq.PUB)
    def test(foo, bar, baz):
        assert isinstance(foo, zmq.Socket), foo
        assert isinstance(bar, zmq.Socket), bar
        assert isinstance(baz, zmq.Socket), baz
        assert foo.context is zmq.Context.instance()
        assert bar.context is zmq.Context.instance()
        assert baz.context is zmq.Context.instance()
        assert foo.type == zmq.PUSH
        assert bar.type == zmq.SUB
        assert baz.type == zmq.PUB
    test()