import errno
import io
import socket
from io import SEEK_END
from typing import Optional, Union
from ..exceptions import ConnectionError, TimeoutError
from ..utils import SSL_AVAILABLE
def purge(self) -> None:
    """
        After a successful read, purge the read part of buffer
        """
    unread = self.unread_bytes()
    if unread > 0:
        return
    if unread > 0:
        view = self._buffer.getbuffer()
        view[:unread] = view[-unread:]
    self._buffer.truncate(unread)
    self._buffer.seek(0)