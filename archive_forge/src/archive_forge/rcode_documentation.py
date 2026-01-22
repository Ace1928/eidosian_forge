import dns.exception
from ._compat import long
Convert rcode into text.

    *value*, and ``int``, the rcode.

    Raises ``ValueError`` if rcode is < 0 or > 4095.

    Returns a ``text``.
    