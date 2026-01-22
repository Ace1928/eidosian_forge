import contextlib
import io
import os
from uuid import uuid4
import requests
from .._compat import fields
def write_to(self, buffer, size):
    """Write the requested amount of bytes to the buffer provided.

        The number of bytes written may exceed size on the first read since we
        load the headers ambitiously.

        :param CustomBytesIO buffer: buffer we want to write bytes to
        :param int size: number of bytes requested to be written to the buffer
        :returns: int -- number of bytes actually written
        """
    written = 0
    if self.headers_unread:
        written += buffer.append(self.headers)
        self.headers_unread = False
    while total_len(self.body) > 0 and (size == -1 or written < size):
        amount_to_read = size
        if size != -1:
            amount_to_read = size - written
        written += buffer.append(self.body.read(amount_to_read))
    return written