from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_is_thenable_then_object():
    promise = FakeThenPromise()
    assert not is_thenable(promise)