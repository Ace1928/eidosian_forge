import sys
from twisted.internet import defer, reactor
from twisted.python import filepath, runtime
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_closeWrite(self):
    """
        Closing a file which was open for writing in a monitored
        directory sends an C{inotify.IN_CLOSE_WRITE} event to the
        callback.
        """

    def operation(path):
        path.open('w').close()
    return self._notificationTest(inotify.IN_CLOSE_WRITE, operation)