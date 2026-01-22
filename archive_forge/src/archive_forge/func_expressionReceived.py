import copy
import struct
from io import BytesIO
from twisted.internet import protocol
from twisted.persisted import styles
from twisted.python import log
from twisted.python.compat import iterbytes
from twisted.python.reflect import fullyQualifiedName
def expressionReceived(self, lst):
    """Called when an expression (list, string, or int) is received."""
    raise NotImplementedError()