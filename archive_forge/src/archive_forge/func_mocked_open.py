import builtins
import struct
from io import StringIO
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.protocols import ident
from twisted.python import failure
from twisted.trial import unittest
def mocked_open(*args, **kwargs):
    """
            Mock for the open call to prevent actually opening /proc/net/tcp.
            """
    open_calls.append((args, kwargs))
    return StringIO(self.sampleFile)