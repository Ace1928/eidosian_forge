import time
from twisted.internet import reactor, task
from twisted.python import failure, log
from twisted.trial import _synctest, reporter, unittest
def test_boundedObservers(self):
    """
        There are no extra log observers after a test runs.
        """
    observer = _synctest._LogObserver()
    self.patch(_synctest, '_logObserver', observer)
    observers = log.theLogPublisher.observers[:]
    test = self.MockTest()
    test(self.result)
    self.assertEqual(observers, log.theLogPublisher.observers)