from zope.interface import implementer
from twisted.internet import interfaces
from twisted.logger import Logger
from twisted.python import log
def requestReceived(self, requestType, data):
    """
        Called when a request is sent to this channel.  By default it delegates
        to self.request_<requestType>.
        If this function returns true, the request succeeded, otherwise it
        failed.

        @type requestType:  L{bytes}
        @type data:         L{bytes}
        @rtype:             L{bool}
        """
    foo = requestType.replace(b'-', b'_').decode('ascii')
    f = getattr(self, 'request_' + foo, None)
    if f:
        return f(data)
    self._log.info('unhandled request for {requestType}', requestType=requestType)
    return 0