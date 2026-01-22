from collections import deque
from twisted.internet.defer import Deferred, TimeoutError, fail
from twisted.protocols.basic import LineReceiver
from twisted.protocols.policies import TimeoutMixin
from twisted.python import log
from twisted.python.compat import nativeString, networkString
def cmd_VALUE(self, line):
    """
        Prepare the reading a value after a get.
        """
    cmd = self._current[0]
    if cmd.command == b'get':
        key, flags, length = line.split()
        cas = b''
    else:
        key, flags, length, cas = line.split()
    self._lenExpected = int(length)
    self._getBuffer = []
    self._bufferLength = 0
    if cmd.multiple:
        if key not in cmd.keys:
            raise RuntimeError('Unexpected commands answer.')
        cmd.currentKey = key
        cmd.values[key] = [int(flags), cas]
    else:
        if cmd.key != key:
            raise RuntimeError('Unexpected commands answer.')
        cmd.flags = int(flags)
        cmd.cas = cas
    self.setRawMode()