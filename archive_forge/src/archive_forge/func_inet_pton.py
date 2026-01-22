import socket
import dns.ipv4
import dns.ipv6
from ._compat import maybe_ord
def inet_pton(family, text):
    """Convert the textual form of a network address into its binary form.

    *family* is an ``int``, the address family.

    *text* is a ``text``, the textual address.

    Raises ``NotImplementedError`` if the address family specified is not
    implemented.

    Returns a ``binary``.
    """
    if family == AF_INET:
        return dns.ipv4.inet_aton(text)
    elif family == AF_INET6:
        return dns.ipv6.inet_aton(text)
    else:
        raise NotImplementedError