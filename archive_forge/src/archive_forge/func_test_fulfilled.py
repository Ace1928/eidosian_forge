from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_fulfilled():
    p = Promise.fulfilled(4)
    assert p.is_fulfilled
    assert p.get() == 4