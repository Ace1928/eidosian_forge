import base64
from enum import Enum, IntEnum
from hyperframe.exceptions import InvalidPaddingError
from hyperframe.frame import (
from hpack.hpack import Encoder, Decoder
from hpack.exceptions import HPACKError, OversizedHeaderListError
from .config import H2Configuration
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .frame_buffer import FrameBuffer
from .settings import Settings, SettingCodes
from .stream import H2Stream, StreamClosedBy
from .utilities import SizeLimitDict, guard_increment_window
from .windows import WindowManager
def clear_outbound_data_buffer(self):
    """
        Clears the outbound data buffer, such that if this call was immediately
        followed by a call to
        :meth:`data_to_send <h2.connection.H2Connection.data_to_send>`, that
        call would return no data.

        This method should not normally be used, but is made available to avoid
        exposing implementation details.
        """
    self._data_to_send = bytearray()