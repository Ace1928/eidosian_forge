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
def recv_stderr_ready(self):
    """
        Returns true if data is buffered and ready to be read from this
        channel's stderr stream.  Only channels using `exec_command` or
        `invoke_shell` without a pty will ever have data on the stderr
        stream.

        :return:
            ``True`` if a `recv_stderr` call on this channel would immediately
            return at least one byte; ``False`` otherwise.

        .. versionadded:: 1.1
        """
    return self.in_stderr_buffer.read_ready()