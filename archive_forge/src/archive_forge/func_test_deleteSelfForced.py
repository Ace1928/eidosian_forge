import sys
from twisted.internet import defer, reactor
from twisted.python import filepath, runtime
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_deleteSelfForced(self):
    """
        Deleting the monitored directory itself sends an
        C{inotify.IN_DELETE_SELF} event to the callback
        even if the mask isn't specified by the call to watch().
        """

    def cbNotified(result):
        watch, filename, events = result
        self.assertEqual(filename.asBytesMode(), self.dirname.asBytesMode())
        self.assertTrue(events & inotify.IN_DELETE_SELF)
    self.inotify.watch(self.dirname, mask=0, callbacks=[lambda *args: d.callback(args)])
    d = defer.Deferred()
    d.addCallback(cbNotified)
    self.dirname.remove()
    return d