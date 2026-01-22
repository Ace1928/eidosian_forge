from promise import Promise
from .utils import assert_exception
from threading import Event
def test_3_2_1():
    """
    Test that the arguments to 'then' are optional.
    """
    p1 = Promise()
    p2 = p1.then()
    p3 = Promise()
    p4 = p3.then()
    p1.do_resolve(5)
    p3.do_reject(Exception('How dare you!'))