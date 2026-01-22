import sys
from functools import partial
from io import BytesIO
from twisted.internet import main, protocol
from twisted.internet.testing import StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.spread import banana
from twisted.trial.unittest import TestCase
def putResult(self, result):
    """
        Store an expression received by C{self.enc}.

        @param result: The object that was received.
        @type result: Any type supported by Banana.
        """
    self.result = result