from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_promise_loop():

    def by_two(result):
        return result * 2

    def executor(resolve, reject):
        resolve(Promise.resolve(1).then(lambda v: Promise.resolve(v).then(by_two)))
    p = Promise(executor)
    assert p.get(0.1) == 2