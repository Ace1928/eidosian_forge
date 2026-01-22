from promise import Promise
from .utils import assert_exception
from threading import Event
def test_3_2_6_5_pending():
    """
    Handles the case where the arguments to then
    are values, not functions or promises.
    """
    p1 = Promise()
    p2 = p1.then(None, 5)
    p1.do_reject(Exception('Error'))
    assert_exception(p1.reason, Exception, 'Error')
    p2._wait()
    assert p2.is_rejected
    assert_exception(p2.reason, Exception, 'Error')