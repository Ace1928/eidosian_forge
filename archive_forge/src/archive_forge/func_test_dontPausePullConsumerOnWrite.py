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
def test_dontPausePullConsumerOnWrite(self):
    """
        Verify that FileDescriptor does not call producer.pauseProducing() on a
        non-streaming pull producer in response to a L{IConsumer.write} call
        which results in a full write buffer. Issue #2286.
        """
    return self._dontPausePullConsumerTest('write')