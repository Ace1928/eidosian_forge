from typing import Type, Union
from twisted.internet.endpoints import (
from twisted.internet.testing import MemoryReactor
from twisted.trial.unittest import SynchronousTestCase as TestCase
from .._parser import unparseEndpoint
from .._wrapper import HAProxyWrappingFactory
def test_colonInKeywordValue(self) -> None:
    """
        Escaped ":" in keyword value.
        """
    self.check('hello=\\:')