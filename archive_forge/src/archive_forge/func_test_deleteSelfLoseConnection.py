import sys
from twisted.internet import defer, reactor
from twisted.python import filepath, runtime
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_deleteSelfLoseConnection(self):
    """
        L{inotify.INotify} closes the file descriptor after removing a
        directory from the filesystem (and therefore from the watchlist).
        """

    def cbNotified(result):

        def _():
            try:
                self.assertFalse(self.inotify._isWatched(self.dirname))
                self.assertFalse(self.inotify.connected)
                d.callback(None)
            except Exception:
                d.errback()
        ignored, filename, events = result
        self.assertEqual(filename.asBytesMode(), self.dirname.asBytesMode())
        self.assertTrue(events & inotify.IN_DELETE_SELF)
        reactor.callLater(0, _)
    self.assertTrue(self.inotify.watch(self.dirname, mask=inotify.IN_DELETE_SELF, callbacks=[lambda *args: notified.callback(args)]))
    notified = defer.Deferred()
    notified.addCallback(cbNotified)
    self.dirname.remove()
    d = defer.Deferred()
    return d