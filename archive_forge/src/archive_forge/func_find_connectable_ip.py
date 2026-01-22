import socket
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union
from selenium.types import AnyKey
from selenium.webdriver.common.keys import Keys
def find_connectable_ip(host: Union[str, bytes, bytearray, None], port: Optional[int]=None) -> Optional[str]:
    """Resolve a hostname to an IP, preferring IPv4 addresses.

    We prefer IPv4 so that we don't change behavior from previous IPv4-only
    implementations, and because some drivers (e.g., FirefoxDriver) do not
    support IPv6 connections.

    If the optional port number is provided, only IPs that listen on the given
    port are considered.

    :Args:
        - host - A hostname.
        - port - Optional port number.

    :Returns:
        A single IP address, as a string. If any IPv4 address is found, one is
        returned. Otherwise, if any IPv6 address is found, one is returned. If
        neither, then None is returned.
    """
    try:
        addrinfos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return None
    ip = None
    for family, _, _, _, sockaddr in addrinfos:
        connectable = True
        if port:
            connectable = is_connectable(port, sockaddr[0])
        if connectable and family == socket.AF_INET:
            return sockaddr[0]
        if connectable and (not ip) and (family == socket.AF_INET6):
            ip = sockaddr[0]
    return ip