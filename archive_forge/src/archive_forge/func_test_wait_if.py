from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_wait_if():
    p1 = Promise()
    p1.do_resolve(5)
    p1._wait()
    assert p1.is_fulfilled