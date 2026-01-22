from io import StringIO
import sys
import dns.exception
import dns.name
import dns.ttl
from ._compat import long, text_type, binary_type
def unget(self, token):
    """Unget a token.

        The unget buffer for tokens is only one token large; it is
        an error to try to unget a token when the unget buffer is not
        empty.

        token: the token to unget

        Raises UngetBufferFull: there is already an ungotten token
        """
    if self.ungotten_token is not None:
        raise UngetBufferFull
    self.ungotten_token = token