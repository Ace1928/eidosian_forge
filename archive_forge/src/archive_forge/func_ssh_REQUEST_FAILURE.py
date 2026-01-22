import string
import struct
import twisted.internet.error
from twisted.conch import error
from twisted.conch.ssh import common, service
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
def ssh_REQUEST_FAILURE(self, packet):
    """
        Our global request failed.  Get the appropriate Deferred and errback
        it with the packet we received.
        """
    self._log.debug('global request failure')
    self.deferreds['global'].pop(0).errback(error.ConchError('global request failed', packet))