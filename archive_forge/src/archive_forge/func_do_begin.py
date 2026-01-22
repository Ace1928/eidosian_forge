from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def do_begin(self, byte):
    if byte.isspace():
        return
    if byte != '<':
        if self.beExtremelyLenient:
            self._leadingBodyData = byte
            return 'bodydata'
        self._parseError(f"First char of document [{byte!r}] wasn't <")
    return 'tagstart'