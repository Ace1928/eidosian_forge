from collections.abc import Sequence as _Sequence
from typing import (
from twisted.python.compat import cmp, comparable
def removeHeader(self, name: AnyStr) -> None:
    """
        Remove the named header from this header object.

        @param name: The name of the HTTP header to remove.

        @return: L{None}
        """
    self._rawHeaders.pop(self._encodeName(name), None)