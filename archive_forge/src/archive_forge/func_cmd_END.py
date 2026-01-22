from collections import deque
from twisted.internet.defer import Deferred, TimeoutError, fail
from twisted.protocols.basic import LineReceiver
from twisted.protocols.policies import TimeoutMixin
from twisted.python import log
from twisted.python.compat import nativeString, networkString
def cmd_END(self):
    """
        This the end token to a get or a stat operation.
        """
    cmd = self._current.popleft()
    if cmd.command == b'get':
        if cmd.multiple:
            values = {key: val[::2] for key, val in cmd.values.items()}
            cmd.success(values)
        else:
            cmd.success((cmd.flags, cmd.value))
    elif cmd.command == b'gets':
        if cmd.multiple:
            cmd.success(cmd.values)
        else:
            cmd.success((cmd.flags, cmd.cas, cmd.value))
    elif cmd.command == b'stats':
        cmd.success(cmd.values)
    else:
        raise RuntimeError('Unexpected END response to {} command'.format(nativeString(cmd.command)))