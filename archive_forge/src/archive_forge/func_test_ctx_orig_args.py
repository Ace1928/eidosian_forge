import threading
from pytest import fixture, raises
import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context
def test_ctx_orig_args():

    @context()
    def f(foo, bar, ctx, baz=None):
        assert isinstance(ctx, zmq.Context), ctx
        assert foo == 42
        assert bar is True
        assert baz == 'mock'
    f(42, True, baz='mock')