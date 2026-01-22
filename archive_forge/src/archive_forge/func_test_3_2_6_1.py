from promise import Promise
from .utils import assert_exception
from threading import Event
def test_3_2_6_1():
    """
    Promises returned by then must be fulfilled when the promise
    they are chained from is fulfilled IF the fulfillment value
    is not a promise.
    """
    p1 = Promise.resolve(5)
    pf = p1.then(lambda v: v * v)
    assert pf.get() == 25
    p2 = Promise.reject(Exception('Error'))
    pr = p2.then(None, lambda r: 5)
    assert 5 == pr.get()