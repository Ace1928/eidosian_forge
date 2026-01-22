from pytest import raises
from promise import Promise
from promise.promise_list import PromiseList
def test_promise_basic():
    all_promises = all([1, 2])
    assert all_promises.get() == [1, 2]