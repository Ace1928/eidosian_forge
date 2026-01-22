import struct
import time
from zope.interface import implementer
from twisted.internet import interfaces, protocol
def getQuote(self):
    """
        Return a quote. May be overrriden in subclasses.
        """
    return b'An apple a day keeps the doctor away.\r\n'