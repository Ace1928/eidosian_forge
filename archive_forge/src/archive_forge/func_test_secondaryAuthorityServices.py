from twisted.internet.base import ThreadedResolver
from twisted.names.client import Resolver
from twisted.names.dns import PORT
from twisted.names.resolve import ResolverChain
from twisted.names.secondary import SecondaryAuthorityService
from twisted.names.tap import Options, _buildResolvers
from twisted.python.runtime import platform
from twisted.python.usage import UsageError
from twisted.trial.unittest import SynchronousTestCase
def test_secondaryAuthorityServices(self) -> None:
    """
        After parsing I{--secondary} options, L{Options} constructs a
        L{SecondaryAuthorityService} instance for each configured secondary.
        """
    options = Options()
    options.parseOptions(['--secondary', '1.2.3.4:5353/example.com', '--secondary', '1.2.3.5:5354/example.com'])
    self.assertEqual(len(options.svcs), 2)
    secondary = options.svcs[0]
    self.assertIsInstance(options.svcs[0], SecondaryAuthorityService)
    self.assertEqual(secondary.primary, '1.2.3.4')
    self.assertEqual(secondary._port, 5353)
    secondary = options.svcs[1]
    self.assertIsInstance(options.svcs[1], SecondaryAuthorityService)
    self.assertEqual(secondary.primary, '1.2.3.5')
    self.assertEqual(secondary._port, 5354)