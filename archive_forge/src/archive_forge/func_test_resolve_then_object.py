from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_resolve_then_object(resolve):
    promise = FakeThenPromise(raises=False)
    p = resolve(promise)
    assert isinstance(p, Promise)