from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_thrown_exceptions_have_stacktrace():

    def throws(v):
        assert False
    p3 = Promise.resolve('a').then(throws)
    with raises(AssertionError) as assert_exc:
        p3.get()
    assert assert_exc.traceback[-1].path.strpath == __file__