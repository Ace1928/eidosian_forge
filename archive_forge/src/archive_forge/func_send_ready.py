import binascii
import os
import socket
import time
import threading
from functools import wraps
from paramiko import util
from paramiko.common import (
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
from paramiko.file import BufferedFile
from paramiko.buffered_pipe import BufferedPipe, PipeTimeout
from paramiko import pipe
from paramiko.util import ClosingContextManager
def send_ready(self):
    """
        Returns true if data can be written to this channel without blocking.
        This means the channel is either closed (so any write attempt would
        return immediately) or there is at least one byte of space in the
        outbound buffer. If there is at least one byte of space in the
        outbound buffer, a `send` call will succeed immediately and return
        the number of bytes actually written.

        :return:
            ``True`` if a `send` call on this channel would immediately succeed
            or fail
        """
    self.lock.acquire()
    try:
        if self.closed or self.eof_sent:
            return True
        return self.out_window_size > 0
    finally:
        self.lock.release()