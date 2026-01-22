from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_fulfill_self():
    p = Promise()
    with raises(TypeError) as excinfo:
        p.do_resolve(p)
        p.get()