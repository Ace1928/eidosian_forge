from twisted.internet import defer, endpoints
from twisted.mail import protocols
from twisted.mail.tap import Options, makeService
from twisted.python.usage import UsageError
from twisted.trial.unittest import TestCase
def test_allProtosDisabledError(self):
    """
        If all protocols are disabled, L{UsageError} is raised.
        """
    options = Options()
    self.assertRaises(UsageError, options.parseOptions, ['--no-pop3', '--no-smtp'])