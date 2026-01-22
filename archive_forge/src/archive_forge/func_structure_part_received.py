from collections import deque
from io import BytesIO
from ... import debug, errors
from ...trace import mutter
def structure_part_received(self, structure):
    if not isinstance(structure, tuple):
        raise errors.SmartProtocolError('Args structure is not a sequence: {!r}'.format(structure))
    if not self._body_started:
        if self.args is not None:
            raise errors.SmartProtocolError('Unexpected structure received: %r (already got %r)' % (structure, self.args))
        self.args = structure
    else:
        if self._body_stream_status != b'E':
            raise errors.SmartProtocolError('Unexpected structure received after body: %r' % (structure,))
        self._body_error_args = structure