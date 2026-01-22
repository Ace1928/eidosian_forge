import time
from twisted.internet import reactor, task
from twisted.python import failure, log
from twisted.trial import _synctest, reporter, unittest
def test_msg(self):
    """
        Test that a standard log message doesn't go anywhere near the result.
        """
    self.observer.gotEvent({'message': ('some message',), 'time': time.time(), 'isError': 0, 'system': '-'})
    self.assertEqual(self.observer.getErrors(), [])