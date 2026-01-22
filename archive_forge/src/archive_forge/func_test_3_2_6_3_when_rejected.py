from promise import Promise
from .utils import assert_exception
from threading import Event
def test_3_2_6_3_when_rejected():
    """
    Testing return of pending promises to make
    sure they are properly chained.
    This covers the case where the root promise
    is rejected after the chaining is defined.
    """
    p1 = Promise()
    pending = Promise()
    pr = p1.then(None, lambda r: pending)
    assert pending.is_pending
    assert pr.is_pending
    p1.do_reject(Exception('Error'))
    pending.do_resolve(10)
    pending._wait()
    assert pending.is_fulfilled
    assert 10 == pending.get()
    assert 10 == pr.get()
    p2 = Promise()
    bad = Promise()
    pr = p2.then(None, lambda r: bad)
    assert bad.is_pending
    assert pr.is_pending
    p2.do_reject(Exception('Error'))
    bad.do_reject(Exception('Assertion'))
    bad._wait()
    assert bad.is_rejected
    assert_exception(bad.reason, Exception, 'Assertion')
    pr._wait()
    assert pr.is_rejected
    assert_exception(pr.reason, Exception, 'Assertion')