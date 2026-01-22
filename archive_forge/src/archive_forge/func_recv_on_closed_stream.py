from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def recv_on_closed_stream(self, previous_state):
    """
        Called when an unexpected frame is received on an already-closed
        stream.

        An endpoint that receives an unexpected frame should treat it as
        a stream error or connection error with type STREAM_CLOSED, depending
        on the specific frame. The error handling is done at a higher level:
        this just raises the appropriate error.
        """
    raise StreamClosedError(self.stream_id)