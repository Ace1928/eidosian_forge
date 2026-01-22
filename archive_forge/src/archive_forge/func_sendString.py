import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
def sendString(self, string):
    """
        Send a prefixed string to the other end of the connection.

        @param string: The string to send.  The necessary framing (length
            prefix, etc) will be added.
        @type string: C{bytes}
        """
    if len(string) >= 2 ** (8 * self.prefixLength):
        raise StringTooLongError('Try to send %s bytes whereas maximum is %s' % (len(string), 2 ** (8 * self.prefixLength)))
    self.transport.write(pack(self.structFormat, len(string)) + string)