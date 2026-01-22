from twisted.internet.base import ThreadedResolver
from twisted.names.client import Resolver
from twisted.names.dns import PORT
from twisted.names.resolve import ResolverChain
from twisted.names.secondary import SecondaryAuthorityService
from twisted.names.tap import Options, _buildResolvers
from twisted.python.runtime import platform
from twisted.python.usage import UsageError
from twisted.trial.unittest import SynchronousTestCase
def test_secondary(self) -> None:
    """
        An argument of the form C{"ip/domain"} is parsed by L{Options} for the
        I{--secondary} option and added to its list of secondaries, using the
        default DNS port number.
        """
    options = Options()
    options.parseOptions(['--secondary', '1.2.3.4/example.com'])
    self.assertEqual([(('1.2.3.4', PORT), ['example.com'])], options.secondaries)