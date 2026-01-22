import socket
import dns.ipv4
import dns.ipv6
from ._compat import maybe_ord
def inet_ntop(family, address):
    """Convert the binary form of a network address into its textual form.

    *family* is an ``int``, the address family.

    *address* is a ``binary``, the network address in binary form.

    Raises ``NotImplementedError`` if the address family specified is not
    implemented.

    Returns a ``text``.
    """
    if family == AF_INET:
        return dns.ipv4.inet_ntoa(address)
    elif family == AF_INET6:
        return dns.ipv6.inet_ntoa(address)
    else:
        raise NotImplementedError