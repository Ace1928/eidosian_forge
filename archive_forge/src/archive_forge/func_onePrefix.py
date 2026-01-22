from typing import Type, Union
from twisted.internet.endpoints import (
from twisted.internet.testing import MemoryReactor
from twisted.trial.unittest import SynchronousTestCase as TestCase
from .._parser import unparseEndpoint
from .._wrapper import HAProxyWrappingFactory
def onePrefix(self, description: str, expectedClass: Union[Type[TCP4ServerEndpoint], Type[TCP6ServerEndpoint], Type[UNIXServerEndpoint]]) -> _WrapperServerEndpoint:
    """
        Test the C{haproxy} enpdoint prefix against one sub-endpoint type.

        @param description: A string endpoint description beginning with
            C{haproxy}.
        @type description: native L{str}

        @param expectedClass: the expected sub-endpoint class given the
            description.
        @type expectedClass: L{type}

        @return: the parsed endpoint
        @rtype: L{IStreamServerEndpoint}

        @raise twisted.trial.unittest.Failtest: if the parsed endpoint doesn't
            match expectations.
        """
    reactor = MemoryReactor()
    endpoint = serverFromString(reactor, description)
    self.assertIsInstance(endpoint, _WrapperServerEndpoint)
    assert isinstance(endpoint, _WrapperServerEndpoint)
    self.assertIsInstance(endpoint._wrappedEndpoint, expectedClass)
    self.assertIs(endpoint._wrapperFactory, HAProxyWrappingFactory)
    return endpoint