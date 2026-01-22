from promise import Promise
from .utils import assert_exception
from threading import Event
def test_3_2_6_3_when_fulfilled():
    """
    Testing return of pending promises to make
    sure they are properly chained.
    This covers the case where the root promise
    is fulfilled after the chaining is defined.
    """
    p1 = Promise()
    pending = Promise()

    def p1_resolved(v):
        return pending
    pf = p1.then(p1_resolved)
    assert pending.is_pending
    assert pf.is_pending
    p1.do_resolve(10)
    pending.do_resolve(5)
    pending._wait()
    assert pending.is_fulfilled
    assert 5 == pending.get()
    pf._wait()
    assert pf.is_fulfilled
    assert 5 == pf.get()
    p2 = Promise()
    bad = Promise()
    pr = p2.then(lambda r: bad)
    assert bad.is_pending
    assert pr.is_pending
    p2.do_resolve(10)
    bad._reject_callback(Exception('Error'))
    bad._wait()
    assert bad.is_rejected
    assert_exception(bad.reason, Exception, 'Error')
    pr._wait()
    assert pr.is_rejected
    assert_exception(pr.reason, Exception, 'Error')