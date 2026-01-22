import sys
from twisted.internet import defer, reactor
from twisted.python import filepath, runtime
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_connectionLostError(self):
    """
        L{inotify.INotify.connectionLost} if there's a problem while closing
        the fd shouldn't raise the exception but should log the error
        """
    import os
    in_ = inotify.INotify()
    os.close(in_._fd)
    in_.loseConnection()
    self.flushLoggedErrors()