import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_runWithConnectionRaiseOriginalError(self):
    """
        If rollback fails, L{ConnectionPool.runWithConnection} raises the
        original exception and log the error of the rollback.
        """

    class ConnectionRollbackRaise:

        def __init__(self, pool):
            pass

        def rollback(self):
            raise RuntimeError('problem!')

    def raisingFunction(connection):
        raise ValueError('foo')
    pool = DummyConnectionPool()
    pool.connectionFactory = ConnectionRollbackRaise
    d = pool.runWithConnection(raisingFunction)
    d = self.assertFailure(d, ValueError)

    def cbFailed(ignored):
        errors = self.flushLoggedErrors(RuntimeError)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].value.args[0], 'problem!')
    d.addCallback(cbFailed)
    return d