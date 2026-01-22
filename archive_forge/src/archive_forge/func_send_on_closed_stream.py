from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def send_on_closed_stream(self, previous_state):
    """
        Called when an attempt is made to send data on an already-closed
        stream.

        This essentially overrides the standard logic by throwing a
        more-specific error: StreamClosedError. This is a ProtocolError, so it
        matches the standard API of the state machine, but provides more detail
        to the user.
        """
    raise StreamClosedError(self.stream_id)