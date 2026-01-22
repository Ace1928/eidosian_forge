from promise import Promise
from .utils import assert_exception
from threading import Event
def test_3_2_6_3_if_fulfilled():
    """
    Testing return of pending promises to make
    sure they are properly chained.
    This covers the case where the root promise
    is fulfilled before the chaining is defined.
    """
    p1 = Promise()
    p1.do_resolve(10)
    pending = Promise()
    pending.do_resolve(5)
    pf = p1.then(lambda r: pending)
    pending._wait()
    assert pending.is_fulfilled
    assert 5 == pending.get()
    pf._wait()
    assert pf.is_fulfilled
    assert 5 == pf.get()
    p2 = Promise()
    p2.do_resolve(10)
    bad = Promise()
    bad.do_reject(Exception('Error'))
    pr = p2.then(lambda r: bad)
    bad._wait()
    assert_exception(bad.reason, Exception, 'Error')
    pr._wait()
    assert pr.is_rejected
    assert_exception(pr.reason, Exception, 'Error')