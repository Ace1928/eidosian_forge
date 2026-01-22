from collections import deque
from twisted.internet.defer import Deferred, TimeoutError, fail
from twisted.protocols.basic import LineReceiver
from twisted.protocols.policies import TimeoutMixin
from twisted.python import log
from twisted.python.compat import nativeString, networkString
def cmd_CLIENT_ERROR(self, errText):
    """
        An invalid input as been sent.
        """
    errText = repr(errText)
    log.err('Invalid input: ' + errText)
    cmd = self._current.popleft()
    cmd.fail(ClientError(errText))