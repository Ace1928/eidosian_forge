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
def push_stream(self, stream_id, promised_stream_id, request_headers):
    """
        Push a response to the client by sending a PUSH_PROMISE frame.

        If it is important to send HPACK "never indexed" header fields (as
        defined in `RFC 7451 Section 7.1.3
        <https://tools.ietf.org/html/rfc7541#section-7.1.3>`_), the user may
        instead provide headers using the HPACK library's :class:`HeaderTuple
        <hpack:hpack.HeaderTuple>` and :class:`NeverIndexedHeaderTuple
        <hpack:hpack.NeverIndexedHeaderTuple>` objects.

        :param stream_id: The ID of the stream that this push is a response to.
        :type stream_id: ``int``
        :param promised_stream_id: The ID of the stream that the pushed
            response will be sent on.
        :type promised_stream_id: ``int``
        :param request_headers: The headers of the request that the pushed
            response will be responding to.
        :type request_headers: An iterable of two tuples of bytestrings or
            :class:`HeaderTuple <hpack:hpack.HeaderTuple>` objects.
        :returns: Nothing
        """
    self.config.logger.debug('Send Push Promise frame on stream ID %d', stream_id)
    if not self.remote_settings.enable_push:
        raise ProtocolError('Remote peer has disabled stream push')
    self.state_machine.process_input(ConnectionInputs.SEND_PUSH_PROMISE)
    stream = self._get_stream_by_id(stream_id)
    if stream_id % 2 == 0:
        raise ProtocolError('Cannot recursively push streams.')
    new_stream = self._begin_new_stream(promised_stream_id, AllowedStreamIDs.EVEN)
    self.streams[promised_stream_id] = new_stream
    frames = stream.push_stream_in_band(promised_stream_id, request_headers, self.encoder)
    new_frames = new_stream.locally_pushed()
    self._prepare_for_sending(frames + new_frames)