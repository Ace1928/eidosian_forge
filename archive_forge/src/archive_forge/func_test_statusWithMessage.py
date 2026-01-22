from twisted.internet.testing import MemoryReactor, StringTransportWithDisconnection
from twisted.trial.unittest import TestCase
from twisted.web.proxy import (
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.test.test_web import DummyRequest
def test_statusWithMessage(self):
    """
        If the response contains a status with a message, it should be
        forwarded to the parent request with all the information.
        """
    return self._testDataForward(404, b'Not Found', [], b'')