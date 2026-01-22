from __future__ import annotations
import logging # isort:skip
import contextlib
import errno
import os
import sys
from typing import TYPE_CHECKING, Iterator
from bokeh.application import Application
from bokeh.application.handlers import (
from bokeh.util.warnings import warn
@contextlib.contextmanager
def report_server_init_errors(address: str | None=None, port: int | None=None, **kwargs: str) -> Iterator[None]:
    """ A context manager to help print more informative error messages when a
    ``Server`` cannot be started due to a network problem.

    Args:
        address (str) : network address that the server will be listening on

        port (int) : network address that the server will be listening on

    Example:

        .. code-block:: python

            with report_server_init_errors(**server_kwargs):
                server = Server(applications, **server_kwargs)

        If there are any errors (e.g. port or address in already in use) then a
        critical error will be logged and the process will terminate with a
        call to ``sys.exit(1)``

    """
    try:
        yield
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            log.critical('Cannot start Bokeh server, port %s is already in use', port)
        elif e.errno == errno.EADDRNOTAVAIL:
            log.critical("Cannot start Bokeh server, address '%s' not available", address)
        else:
            codename = errno.errorcode[e.errno]
            log.critical('Cannot start Bokeh server [%s]: %r', codename, e)
        sys.exit(1)