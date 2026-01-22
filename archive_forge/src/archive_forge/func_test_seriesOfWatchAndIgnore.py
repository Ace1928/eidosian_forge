import sys
from twisted.internet import defer, reactor
from twisted.python import filepath, runtime
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_seriesOfWatchAndIgnore(self):
    """
        L{inotify.INotify} will watch a filepath for events even if the same
        path is repeatedly added/removed/re-added to the watchpoints.
        """
    expectedPath = self.dirname.child('foo.bar2')
    expectedPath.touch()
    notified = defer.Deferred()

    def cbNotified(result):
        ignored, filename, events = result
        self.assertEqual(filename.asBytesMode(), expectedPath.asBytesMode())
        self.assertTrue(events & inotify.IN_DELETE_SELF)

    def callIt(*args):
        notified.callback(args)
    self.assertTrue(self.inotify.watch(expectedPath, callbacks=[callIt]))
    self.inotify.ignore(expectedPath)
    self.assertTrue(self.inotify.watch(expectedPath, mask=inotify.IN_DELETE_SELF, callbacks=[callIt]))
    notified.addCallback(cbNotified)
    expectedPath.remove()
    return notified