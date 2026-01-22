import socket
from typing import Any, Optional, Tuple
import dns.ipv4
import dns.ipv6
def is_address(text: str) -> bool:
    """Is the specified string an IPv4 or IPv6 address?

    *text*, a ``str``, the textual address.

    Returns a ``bool``.
    """
    try:
        dns.ipv4.inet_aton(text)
        return True
    except Exception:
        try:
            dns.ipv6.inet_aton(text, True)
            return True
        except Exception:
            return False