from unittest import skipIf
from twisted.internet import defer, protocol, reactor
from twisted.internet.task import deferLater
from twisted.trial import unittest, util
def test_exceptionGreaterThan64kEncoded(self) -> None:
    """
        A test which synchronously raises an exception with a long string
        representation including non-ascii content.
        """
    raise Exception('☃' * 2 ** 15)