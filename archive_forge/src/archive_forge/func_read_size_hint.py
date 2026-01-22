import re
from io import BytesIO
from .. import errors
def read_size_hint(self):
    hint = 16384
    if self._state_handler == self._state_expecting_body:
        remaining = self._current_record_length - len(self._buffer)
        if remaining < 0:
            remaining = 0
        return max(hint, remaining)
    return hint