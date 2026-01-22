import errno
import io
import socket
from io import SEEK_END
from typing import Optional, Union
from ..exceptions import ConnectionError, TimeoutError
from ..utils import SSL_AVAILABLE
def rewind(self, pos: int) -> None:
    """
        Rewind the buffer to a specific position, to re-start reading
        """
    self._buffer.seek(pos)