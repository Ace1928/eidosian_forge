import logging
import logging.handlers
import os
import socket
import sys
from humanfriendly import coerce_boolean
from humanfriendly.compat import on_macos, on_windows
from coloredlogs import (
def connect_to_syslog(address=None, facility=None, level=None):
    """
    Create a :class:`~logging.handlers.SysLogHandler`.

    :param address: The device file or network address of the system logging
                    daemon (a string or tuple, defaults to the result of
                    :func:`find_syslog_address()`).
    :param facility: Refer to :class:`~logging.handlers.SysLogHandler`.
                     Defaults to ``LOG_USER``.
    :param level: The logging level for the :class:`~logging.handlers.SysLogHandler`
                  (defaults to :data:`.DEFAULT_LOG_LEVEL`). This value is coerced
                  using :func:`~coloredlogs.level_to_number()`.
    :returns: A :class:`~logging.handlers.SysLogHandler` object or :data:`None` (if the
              system logging daemon is unavailable).

    The process of connecting to the system logging daemon goes as follows:

    - The following two socket types are tried (in decreasing preference):

       1. :data:`~socket.SOCK_RAW` avoids truncation of log messages but may
          not be supported.
       2. :data:`~socket.SOCK_STREAM` (TCP) supports longer messages than the
          default (which is UDP).
    """
    if not address:
        address = find_syslog_address()
    if facility is None:
        facility = logging.handlers.SysLogHandler.LOG_USER
    if level is None:
        level = DEFAULT_LOG_LEVEL
    for socktype in (socket.SOCK_RAW, socket.SOCK_STREAM, None):
        kw = dict(facility=facility, address=address)
        if socktype is not None:
            kw['socktype'] = socktype
        try:
            handler = logging.handlers.SysLogHandler(**kw)
        except IOError:
            pass
        else:
            handler.setLevel(level_to_number(level))
            return handler