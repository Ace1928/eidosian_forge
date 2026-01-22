from __future__ import absolute_import
import contextlib
import socket
import google_auth_oauthlib.flow
def find_open_port(start=8080, stop=None):
    """Find an open port between ``start`` and ``stop``.
    Parameters
    ----------
    start : Optional[int]
        Beginning of range of ports to try. Defaults to 8080.
    stop : Optional[int]
        End of range of ports to try (not including exactly equals ``stop``).
        This function tries 100 possible ports if no ``stop`` is specified.
    Returns
    -------
    Optional[int]
        ``None`` if no open port is found, otherwise an integer indicating an
        open port.
    """
    if not stop:
        stop = start + DEFAULT_PORTS_TO_TRY
    for port in range(start, stop):
        if is_port_open(port):
            return port
    return None