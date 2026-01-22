import sys
from functools import partial
from io import BytesIO
from twisted.internet import main, protocol
from twisted.internet.testing import StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.spread import banana
from twisted.trial.unittest import TestCase
def test_unsupportedUserType(self):
    """
        Banana does not support arbitrary user-defined types (such as those
        defined with the ``class`` statement).  ``Banana.sendEncoded`` raises
        ``BananaError`` if called with an instance of such a type.
        """
    self._unsupportedTypeTest(MathTests(), __name__ + '.MathTests')