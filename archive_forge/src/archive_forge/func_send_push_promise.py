from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def send_push_promise(self, previous_state):
    """
        Fires on the already-existing stream when a PUSH_PROMISE frame is sent.
        We may only send PUSH_PROMISE frames if we're a server.
        """
    if self.client is True:
        raise ProtocolError('Cannot push streams from client peers.')
    event = _PushedRequestSent()
    return [event]