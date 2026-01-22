from promise import Promise
from .utils import assert_exception
from threading import Event
def test_promise_follows_indifentely():
    a = Promise.resolve(None)
    b = a.then(lambda x: Promise.resolve('X'))
    e = Event()

    def b_then(v):
        c = Promise.resolve(None)
        d = c.then(lambda v: Promise.resolve('B'))
        return d
    promise = b.then(b_then)
    assert promise.get() == 'B'