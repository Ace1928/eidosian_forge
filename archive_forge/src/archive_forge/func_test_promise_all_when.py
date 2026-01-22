from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_promise_all_when():
    p1 = Promise()
    p2 = Promise()
    pl = Promise.all([p1, p2])
    assert p1.is_pending
    assert p2.is_pending
    assert pl.is_pending
    p1.do_resolve(5)
    p1._wait()
    assert p1.is_fulfilled
    assert p2.is_pending
    assert pl.is_pending
    p2.do_resolve(10)
    p2._wait()
    pl._wait()
    assert p1.is_fulfilled
    assert p2.is_fulfilled
    assert pl.is_fulfilled
    assert 5 == p1.get()
    assert 10 == p2.get()
    assert 5 == pl.get()[0]
    assert 10 == pl.get()[1]