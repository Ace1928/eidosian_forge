import errno
import io
import socket
from io import SEEK_END
from typing import Optional, Union
from ..exceptions import ConnectionError, TimeoutError
from ..utils import SSL_AVAILABLE
def get_pos(self) -> int:
    """
        Get current read position
        """
    return self._buffer.tell()