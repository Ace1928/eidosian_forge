from collections import deque
from io import BytesIO
from ... import debug, errors
from ...trace import mutter
def protocol_error(self, exception):
    self.finished_reading = True
    self._medium_request.finished_reading()
    raise