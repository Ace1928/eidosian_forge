import threading
from pytest import fixture, raises
import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context
def test_ctx_kw_naming():

    @context(name='myctx')
    def test(myctx):
        assert isinstance(myctx, zmq.Context), myctx
    test()