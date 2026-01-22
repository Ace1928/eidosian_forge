from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
def get_tail(self):
    """Read back any unused data."""
    return self._readahead.getvalue()