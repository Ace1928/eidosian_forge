import collections
import sys
import warnings
from . import protocols
from . import transports
from .log import logger
@property
def need_ssldata(self):
    """Whether more record level data is needed to complete a handshake
        that is currently in progress."""
    return self._need_ssldata