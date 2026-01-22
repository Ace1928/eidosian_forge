import binascii
import dns.name
import dns.ipv6
import dns.ipv4
from dns._compat import PY3
def to_address(name):
    """Convert a reverse map domain name into textual address form.

    *name*, a ``dns.name.Name``, an IPv4 or IPv6 address in reverse-map name
    form.

    Raises ``dns.exception.SyntaxError`` if the name does not have a
    reverse-map form.

    Returns a ``text``.
    """
    if name.is_subdomain(ipv4_reverse_domain):
        name = name.relativize(ipv4_reverse_domain)
        labels = list(name.labels)
        labels.reverse()
        text = b'.'.join(labels)
        return dns.ipv4.inet_ntoa(dns.ipv4.inet_aton(text))
    elif name.is_subdomain(ipv6_reverse_domain):
        name = name.relativize(ipv6_reverse_domain)
        labels = list(name.labels)
        labels.reverse()
        parts = []
        i = 0
        l = len(labels)
        while i < l:
            parts.append(b''.join(labels[i:i + 4]))
            i += 4
        text = b':'.join(parts)
        return dns.ipv6.inet_ntoa(dns.ipv6.inet_aton(text))
    else:
        raise dns.exception.SyntaxError('unknown reverse-map address family')