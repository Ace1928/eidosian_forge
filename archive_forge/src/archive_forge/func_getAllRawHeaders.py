from collections.abc import Sequence as _Sequence
from typing import (
from twisted.python.compat import cmp, comparable
def getAllRawHeaders(self) -> Iterator[Tuple[bytes, Sequence[bytes]]]:
    """
        Return an iterator of key, value pairs of all headers contained in this
        object, as L{bytes}.  The keys are capitalized in canonical
        capitalization.
        """
    for k, v in self._rawHeaders.items():
        yield (self._canonicalNameCaps(k), v)