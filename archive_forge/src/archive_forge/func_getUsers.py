import struct
import time
from zope.interface import implementer
from twisted.internet import interfaces, protocol
def getUsers(self):
    """
        Return active users. Override in subclasses.
        """
    return b'root\r\n'