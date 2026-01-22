from collections import deque
from io import BytesIO
from ... import debug, errors
from ...trace import mutter
def byte_part_received(self, byte):
    if not isinstance(byte, bytes):
        raise TypeError(byte)
    if byte not in [b'E', b'S']:
        raise errors.SmartProtocolError('Unknown response status: {!r}'.format(byte))
    if self._body_started:
        if self._body_stream_status is not None:
            raise errors.SmartProtocolError('Unexpected byte part received: {!r}'.format(byte))
        self._body_stream_status = byte
    else:
        if self.status is not None:
            raise errors.SmartProtocolError('Unexpected byte part received: {!r}'.format(byte))
        self.status = byte