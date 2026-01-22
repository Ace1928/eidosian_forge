from twisted.internet.base import ThreadedResolver
from twisted.names.client import Resolver
from twisted.names.dns import PORT
from twisted.names.resolve import ResolverChain
from twisted.names.secondary import SecondaryAuthorityService
from twisted.names.tap import Options, _buildResolvers
from twisted.python.runtime import platform
from twisted.python.usage import UsageError
from twisted.trial.unittest import SynchronousTestCase
def test_recursiveConfiguration(self) -> None:
    """
        Recursive DNS lookups, if enabled, should be a last-resort option.
        Any other lookup method (cache, local lookup, etc.) should take
        precedence over recursive lookups
        """
    options = Options()
    options.parseOptions(['--hosts-file', 'hosts.txt', '--recursive'])
    ca, cl = _buildResolvers(options)
    for x in cl:
        if isinstance(x, ResolverChain):
            recurser = x.resolvers[-1]
            if isinstance(recurser, Resolver):
                recurser._parseCall.cancel()
    if platform.getType() != 'posix':
        from twisted.internet import reactor
        for x in reactor._newTimedCalls:
            self.assertEqual(x.func.__func__, ThreadedResolver._cleanup)
            x.cancel()
    self.assertIsInstance(cl[-1], ResolverChain)