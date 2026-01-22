import re
from hashlib import md5
from typing import List
from twisted.internet import defer, error, interfaces
from twisted.mail._except import (
from twisted.protocols import basic, policies
from twisted.python import log
def listSize(self, consumer=None):
    """
        Send a LIST command to retrieve the sizes of all messages on the
        server.

        @type consumer: L{None} or callable that takes
            2-L{tuple} of (0) L{int}, (1) L{int}
        @param consumer: A function which consumes the 0-based message index
            and message size derived from the server response.

        @rtype: L{Deferred <defer.Deferred>} which fires L{list} of L{int} or
            callable that takes 2-L{tuple} of (0) L{int}, (1) L{int}
        @return: A deferred which fires when the entire response has been
            received.  When a consumer is not provided, the return value is a
            list of message sizes.  Otherwise, it returns the consumer itself.
        """
    return self._consumeOrSetItem(b'LIST', None, consumer, _listXform)