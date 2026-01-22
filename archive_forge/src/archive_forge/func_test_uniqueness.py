from twisted.trial.unittest import TestCase
from twisted.internet.defer import Deferred, fail, succeed
from .._resultstore import ResultStore
from .._eventloop import EventualResult
def test_uniqueness(self):
    """
        Each store() operation returns a larger number, ensuring uniqueness.
        """
    store = ResultStore()
    dr = EventualResult(Deferred(), None)
    previous = store.store(dr)
    for i in range(100):
        store.retrieve(previous)
        dr = EventualResult(Deferred(), None)
        uid = store.store(dr)
        self.assertTrue(uid > previous)
        previous = uid