from io import StringIO
import sys
import dns.exception
import dns.name
import dns.ttl
from ._compat import long, text_type, binary_type
def skip_whitespace(self):
    """Consume input until a non-whitespace character is encountered.

        The non-whitespace character is then ungotten, and the number of
        whitespace characters consumed is returned.

        If the tokenizer is in multiline mode, then newlines are whitespace.

        Returns the number of characters skipped.
        """
    skipped = 0
    while True:
        c = self._get_char()
        if c != ' ' and c != '\t':
            if c != '\n' or not self.multiline:
                self._unget_char(c)
                return skipped
        skipped += 1