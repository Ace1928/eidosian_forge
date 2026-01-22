from io import StringIO
import sys
import dns.exception
import dns.name
import dns.ttl
from ._compat import long, text_type, binary_type
def get_ttl(self):
    """Read the next token and interpret it as a DNS TTL.

        Raises dns.exception.SyntaxError or dns.ttl.BadTTL if not an
        identifier or badly formed.

        Returns an int.
        """
    token = self.get().unescape()
    if not token.is_identifier():
        raise dns.exception.SyntaxError('expecting an identifier')
    return dns.ttl.from_text(token.value)