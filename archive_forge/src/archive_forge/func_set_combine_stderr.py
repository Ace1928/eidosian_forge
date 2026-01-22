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
def set_combine_stderr(self, combine):
    """
        Set whether stderr should be combined into stdout on this channel.
        The default is ``False``, but in some cases it may be convenient to
        have both streams combined.

        If this is ``False``, and `exec_command` is called (or ``invoke_shell``
        with no pty), output to stderr will not show up through the `recv`
        and `recv_ready` calls.  You will have to use `recv_stderr` and
        `recv_stderr_ready` to get stderr output.

        If this is ``True``, data will never show up via `recv_stderr` or
        `recv_stderr_ready`.

        :param bool combine:
            ``True`` if stderr output should be combined into stdout on this
            channel.
        :return: the previous setting (a `bool`).

        .. versionadded:: 1.1
        """
    data = bytes()
    self.lock.acquire()
    try:
        old = self.combine_stderr
        self.combine_stderr = combine
        if combine and (not old):
            data = self.in_stderr_buffer.empty()
    finally:
        self.lock.release()
    if len(data) > 0:
        self._feed(data)
    return old