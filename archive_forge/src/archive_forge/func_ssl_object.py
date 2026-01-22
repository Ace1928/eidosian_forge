import collections
import sys
import warnings
from . import protocols
from . import transports
from .log import logger
@property
def ssl_object(self):
    """The internal ssl.SSLObject instance.

        Return None if the pipe is not wrapped.
        """
    return self._sslobj