from collections.abc import Sequence as _Sequence
from typing import (
from twisted.python.compat import cmp, comparable
def getRawHeaders(self, name: AnyStr, default: Optional[_T]=None) -> Union[Sequence[AnyStr], Optional[_T]]:
    """
        Returns a sequence of headers matching the given name as the raw string
        given.

        @param name: The name of the HTTP header to get the values of.

        @param default: The value to return if no header with the given C{name}
            exists.

        @return: If the named header is present, a sequence of its
            values.  Otherwise, C{default}.
        """
    encodedName = self._encodeName(name)
    values = self._rawHeaders.get(encodedName, [])
    if not values:
        return default
    if isinstance(name, str):
        return [v.decode('utf8') for v in values]
    return values