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
def increment_flow_control_window(self, increment, stream_id=None):
    """
        Increment a flow control window, optionally for a single stream. Allows
        the remote peer to send more data.

        .. versionchanged:: 2.0.0
           Rejects attempts to increment the flow control window by out of
           range values with a ``ValueError``.

        :param increment: The amount to increment the flow control window by.
        :type increment: ``int``
        :param stream_id: (optional) The ID of the stream that should have its
            flow control window opened. If not present or ``None``, the
            connection flow control window will be opened instead.
        :type stream_id: ``int`` or ``None``
        :returns: Nothing
        :raises: ``ValueError``
        """
    if not 1 <= increment <= self.MAX_WINDOW_INCREMENT:
        raise ValueError('Flow control increment must be between 1 and %d' % self.MAX_WINDOW_INCREMENT)
    self.state_machine.process_input(ConnectionInputs.SEND_WINDOW_UPDATE)
    if stream_id is not None:
        stream = self.streams[stream_id]
        frames = stream.increase_flow_control_window(increment)
        self.config.logger.debug('Increase stream ID %d flow control window by %d', stream_id, increment)
    else:
        self._inbound_flow_control_window_manager.window_opened(increment)
        f = WindowUpdateFrame(0)
        f.window_increment = increment
        frames = [f]
        self.config.logger.debug('Increase connection flow control window by %d', increment)
    self._prepare_for_sending(frames)