import os
import sys
import time
from unittest import skipIf
from twisted.internet import abstract, base, defer, error, interfaces, protocol, reactor
from twisted.internet.defer import Deferred, passthru
from twisted.internet.tcp import Connector
from twisted.python import util
from twisted.trial.unittest import TestCase
import %(reactor)s
from twisted.internet import reactor
def test_scheduledThreadsNotRunUntilReactorRuns(self):
    """
        Scheduled tasks should not be run until the reactor starts running.
        """

    def incAndFinish():
        self.counter = 1
        self.deferred.callback(True)
    self.schedule(incAndFinish)
    self.assertEqual(self.counter, 0)
    return self.deferred