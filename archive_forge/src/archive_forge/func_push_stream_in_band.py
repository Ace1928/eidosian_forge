from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def push_stream_in_band(self, related_stream_id, headers, encoder):
    """
        Returns a list of PUSH_PROMISE/CONTINUATION frames to emit as a pushed
        stream header. Called on the stream that has the PUSH_PROMISE frame
        sent on it.
        """
    self.config.logger.debug('Push stream %r', self)
    events = self.state_machine.process_input(StreamInputs.SEND_PUSH_PROMISE)
    ppf = PushPromiseFrame(self.stream_id)
    ppf.promised_stream_id = related_stream_id
    hdr_validation_flags = self._build_hdr_validation_flags(events)
    frames = self._build_headers_frames(headers, encoder, ppf, hdr_validation_flags)
    return frames