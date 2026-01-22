import itertools
import os
import sys
from unittest import skipIf
from twisted.internet import defer, error, protocol, reactor, stdio
from twisted.python import filepath, log
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.test.test_tcp import ConnectionLostNotifyingProtocol
from twisted.trial.unittest import SkipTest, TestCase
@skipIf(platform.isWindows(), 'StandardIO does not accept stdout as an argument to Windows.  Testing redirection to a file is therefore harder.')
def test_normalFileStandardOut(self):
    """
        If L{StandardIO} is created with a file descriptor which refers to a
        normal file (ie, a file from the filesystem), L{StandardIO.write}
        writes bytes to that file.  In particular, it does not immediately
        consider the file closed or call its protocol's C{connectionLost}
        method.
        """
    onConnLost = defer.Deferred()
    proto = ConnectionLostNotifyingProtocol(onConnLost)
    path = filepath.FilePath(self.mktemp())
    self.normal = normal = path.open('wb')
    self.addCleanup(normal.close)
    kwargs = dict(stdout=normal.fileno())
    if not platform.isWindows():
        r, w = os.pipe()
        self.addCleanup(os.close, r)
        self.addCleanup(os.close, w)
        kwargs['stdin'] = r
    connection = stdio.StandardIO(proto, **kwargs)
    howMany = 5
    count = itertools.count()

    def spin():
        for value in count:
            if value == howMany:
                connection.loseConnection()
                return
            connection.write(b'%d' % (value,))
            break
        reactor.callLater(0, spin)
    reactor.callLater(0, spin)

    def cbLost(reason):
        self.assertEqual(next(count), howMany + 1)
        self.assertEqual(path.getContent(), b''.join((b'%d' % (i,) for i in range(howMany))))
    onConnLost.addCallback(cbLost)
    return onConnLost