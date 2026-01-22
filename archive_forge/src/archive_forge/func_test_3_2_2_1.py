from promise import Promise
from .utils import assert_exception
from threading import Event
def test_3_2_2_1():
    """
    The first argument to 'then' must be called when a promise is
    fulfilled.
    """
    c = Counter()

    def check(v, c):
        assert v == 5
        c.tick()
    p1 = Promise.resolve(5)
    p2 = p1.then(lambda v: check(v, c))
    p2._wait()
    assert 1 == c.value()