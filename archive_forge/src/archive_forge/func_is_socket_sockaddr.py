from socket import AF_UNSPEC as _AF_UNSPEC
from ._daemon import (__version__,
def is_socket_sockaddr(fileobj, address, type=0, flowinfo=0, listening=-1):
    """Check socket type, address and/or port, flowinfo, listening state.

    Wraps sd_is_socket_inet_sockaddr(3).

    `address` is a systemd-style numerical IPv4 or IPv6 address as used in
    ListenStream=. A port may be included after a colon (":").
    See systemd.socket(5) for details.

    Constants for `family` are defined in the socket module.
    """
    fd = _convert_fileobj(fileobj)
    return _is_socket_sockaddr(fd, address, type, flowinfo, listening)