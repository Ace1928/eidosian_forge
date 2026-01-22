import sys
from twisted.internet import defer, reactor
from twisted.python import filepath, runtime
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_closeNoWrite(self):
    """
        Closing a file which was open for reading but not writing in a
        monitored directory sends an C{inotify.IN_CLOSE_NOWRITE} event
        to the callback.
        """

    def operation(path):
        path.touch()
        path.open('r').close()
    return self._notificationTest(inotify.IN_CLOSE_NOWRITE, operation)