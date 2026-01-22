from unittest import skipIf
from twisted.internet import defer, protocol, reactor
from twisted.internet.task import deferLater
from twisted.trial import unittest, util
def test_exceptionGreaterThan64k(self) -> None:
    """
        A test which raises an exception with a long string representation
        synchronously.
        """
    raise LargeError(2 ** 16)