import copy
import os
import pickle
from io import StringIO
from unittest import skipIf
from twisted.application import app, internet, reactors, service
from twisted.application.internet import backoffPolicy
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.testing import MemoryReactor
from twisted.persisted import sob
from twisted.plugins import twisted_reactors
from twisted.protocols import basic, wire
from twisted.python import usage
from twisted.python.runtime import platformType
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SkipTest, TestCase
def test_does_not_overflow_with_large_factor_value(self):
    """
        Even with unusual parameters, any L{OverflowError} within
        L{backoffPolicy()} will be caught and L{maxDelay} will be returned
        instead
        """
    pol = backoffPolicy(1.0, 60.0, 10000000000.0, jitter=lambda: 1)
    self.assertEqual(pol(1751), 61)