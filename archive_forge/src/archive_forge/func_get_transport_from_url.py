import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def get_transport_from_url(url, possible_transports=None):
    """Open a transport to access a URL.

    Args:
      base: a URL
      transports: optional reusable transports list. If not None, created
        transports will be added to the list.

    Returns: A new transport optionally sharing its connection with one of
        possible_transports.
    """
    transport = None
    if possible_transports is not None:
        for t in possible_transports:
            t_same_connection = t._reuse_for(url)
            if t_same_connection is not None:
                if t_same_connection not in possible_transports:
                    possible_transports.append(t_same_connection)
                return t_same_connection
    last_err = None
    for proto, factory_list in transport_list_registry.items():
        if proto is not None and url.startswith(proto):
            transport, last_err = _try_transport_factories(url, factory_list)
            if transport:
                if possible_transports is not None:
                    if transport in possible_transports:
                        raise AssertionError()
                    possible_transports.append(transport)
                return transport
    if not urlutils.is_url(url):
        raise urlutils.InvalidURL(path=url)
    raise UnsupportedProtocol(url, last_err)