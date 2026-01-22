from promise import Promise
from .utils import assert_exception
from threading import Event
def test_3_2_6_4_fulfilled():
    """
    Handles the case where the arguments to then
    are values, not functions or promises.
    """
    p1 = Promise()
    p1.do_resolve(10)
    p2 = p1.then(5)
    assert 10 == p1.get()
    p2._wait()
    assert p2.is_fulfilled
    assert 10 == p2.get()