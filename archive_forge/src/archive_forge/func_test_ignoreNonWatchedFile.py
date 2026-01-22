import sys
from twisted.internet import defer, reactor
from twisted.python import filepath, runtime
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_ignoreNonWatchedFile(self):
    """
        L{inotify.INotify} will raise KeyError if a non-watched filepath is
        ignored.
        """
    expectedPath = self.dirname.child('foo.ignored')
    expectedPath.touch()
    self.assertRaises(KeyError, self.inotify.ignore, expectedPath)