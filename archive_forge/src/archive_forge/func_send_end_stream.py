from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def send_end_stream(self, previous_state):
    """
        Called when an attempt is made to send END_STREAM in the
        HALF_CLOSED_REMOTE state.
        """
    self.stream_closed_by = StreamClosedBy.SEND_END_STREAM