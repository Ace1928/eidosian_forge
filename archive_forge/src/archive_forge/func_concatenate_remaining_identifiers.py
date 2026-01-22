import io
import sys
from typing import Any, List, Optional, Tuple
import dns.exception
import dns.name
import dns.ttl
def concatenate_remaining_identifiers(self, allow_empty: bool=False) -> str:
    """Read the remaining tokens on the line, which should be identifiers.

        Raises dns.exception.SyntaxError if there are no remaining tokens,
        unless `allow_empty=True` is given.

        Raises dns.exception.SyntaxError if a token is seen that is not an
        identifier.

        Returns a string containing a concatenation of the remaining
        identifiers.
        """
    s = ''
    while True:
        token = self.get().unescape()
        if token.is_eol_or_eof():
            self.unget(token)
            break
        if not token.is_identifier():
            raise dns.exception.SyntaxError
        s += token.value
    if not (allow_empty or s):
        raise dns.exception.SyntaxError('expecting another identifier')
    return s