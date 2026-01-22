from twisted.internet import defer, endpoints
from twisted.mail import protocols
from twisted.mail.tap import Options, makeService
from twisted.python.usage import UsageError
from twisted.trial.unittest import TestCase
def test_endpointSMTP(self):
    """
        When I{--smtp} is given a TCP endpoint description as an argument, a
        TCPServerEndpoint is added to the list of SMTP endpoints.
        """
    self._endpointTest('smtp')