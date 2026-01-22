from collections import deque
from io import BytesIO
from ... import debug, errors
from ...trace import mutter
def read_streamed_body(self):
    while not self.finished_reading:
        while self._bytes_parts:
            bytes_part = self._bytes_parts.popleft()
            if 'hpssdetail' in debug.debug_flags:
                mutter('              %d byte part read', len(bytes_part))
            yield bytes_part
        self._read_more()
    if self._body_stream_status == b'E':
        _raise_smart_server_error(self._body_error_args)