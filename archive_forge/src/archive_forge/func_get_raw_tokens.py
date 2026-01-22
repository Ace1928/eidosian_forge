import codecs
import re
import unicodedata
from abc import ABC, ABCMeta
from typing import Iterator, Tuple, Sequence, Any, NamedTuple
def get_raw_tokens(self, chars: str, final: bool=False) -> Iterator[Token]:
    """Yield tokens without any further processing. Tokens are one of:

        - ``\\<word>``: a control word (i.e. a command)
        - ``\\<symbol>``: a control symbol (i.e. \\^ etc.)
        - ``#<n>``: a parameter
        - a series of byte characters
        """
    if self.raw_buffer.text:
        chars = self.raw_buffer.text + chars
    self.raw_buffer = self.emptytoken
    for match in self.regexp.finditer(chars):
        if self.raw_buffer.text:
            yield self.raw_buffer
        assert match.lastgroup is not None
        self.raw_buffer = Token(match.lastgroup, match.group(0))
    if final:
        for token in self.flush_raw_tokens():
            yield token