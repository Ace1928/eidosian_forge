from promise import Promise
from .utils import assert_exception
from threading import Event
def test_promise_all_follows_indifentely():
    promises = Promise.all([Promise.resolve('A'), Promise.resolve(None).then(Promise.resolve).then(lambda v: Promise.resolve(None).then(lambda v: Promise.resolve('B')))])
    assert promises.get() == ['A', 'B']