import collections
import socket
import sys
import warnings
import weakref
from . import coroutines
from . import events
from . import exceptions
from . import format_helpers
from . import protocols
from .log import logger
from .tasks import sleep
def set_transport(self, transport):
    assert self._transport is None, 'Transport already set'
    self._transport = transport