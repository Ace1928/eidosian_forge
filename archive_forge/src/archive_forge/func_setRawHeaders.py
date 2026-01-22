from collections.abc import Sequence as _Sequence
from typing import (
from twisted.python.compat import cmp, comparable
def setRawHeaders(self, name: Union[str, bytes], values: object) -> None:
    """
        Sets the raw representation of the given header.

        @param name: The name of the HTTP header to set the values for.

        @param values: A list of strings each one being a header value of
            the given name.

        @raise TypeError: Raised if C{values} is not a sequence of L{bytes}
            or L{str}, or if C{name} is not L{bytes} or L{str}.

        @return: L{None}
        """
    if not isinstance(values, _Sequence):
        raise TypeError('Header entry %r should be sequence but found instance of %r instead' % (name, type(values)))
    if not isinstance(name, (bytes, str)):
        raise TypeError(f'Header name is an instance of {type(name)!r}, not bytes or str')
    for count, value in enumerate(values):
        if not isinstance(value, (bytes, str)):
            raise TypeError('Header value at position %s is an instance of %r, not bytes or str' % (count, type(value)))
    _name = _sanitizeLinearWhitespace(self._encodeName(name))
    encodedValues: List[bytes] = []
    for v in values:
        if isinstance(v, str):
            _v = v.encode('utf8')
        else:
            _v = v
        encodedValues.append(_sanitizeLinearWhitespace(_v))
    self._rawHeaders[_name] = encodedValues