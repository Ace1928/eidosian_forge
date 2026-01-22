import sys
from functools import partial
from io import BytesIO
from twisted.internet import main, protocol
from twisted.internet.testing import StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.spread import banana
from twisted.trial.unittest import TestCase
def selectDialect(protocol, dialect):
    """
    Dictate a Banana dialect to use.

    @param protocol: A L{banana.Banana} instance which has not yet had a
        dialect negotiated.

    @param dialect: A L{bytes} instance naming a Banana dialect to select.
    """
    protocol._selectDialect(dialect)