import binascii
import dns.name
import dns.ipv6
import dns.ipv4
from dns._compat import PY3
Convert a reverse map domain name into textual address form.

    *name*, a ``dns.name.Name``, an IPv4 or IPv6 address in reverse-map name
    form.

    Raises ``dns.exception.SyntaxError`` if the name does not have a
    reverse-map form.

    Returns a ``text``.
    