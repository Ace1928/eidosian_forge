import threading
from pytest import fixture, raises
import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context
def test_ctx_arg_kwarg():

    @context('ctx', io_threads=5)
    def test(ctx):
        assert isinstance(ctx, zmq.Context), ctx
        assert ctx.IO_THREADS == 5, ctx.IO_THREADS
    test()