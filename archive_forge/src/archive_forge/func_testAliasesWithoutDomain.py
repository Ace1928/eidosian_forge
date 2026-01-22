from twisted.internet import defer, endpoints
from twisted.mail import protocols
from twisted.mail.tap import Options, makeService
from twisted.python.usage import UsageError
from twisted.trial.unittest import TestCase
def testAliasesWithoutDomain(self):
    """
        Test that adding an aliases(5) file before adding a domain raises a
        UsageError.
        """
    self.assertRaises(UsageError, Options().parseOptions, ['--aliases', self.aliasFilename])