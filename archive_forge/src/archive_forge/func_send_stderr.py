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
def send_stderr(self, s):
    """
        Send data to the channel on the "stderr" stream.  This is normally
        only used by servers to send output from shell commands -- clients
        won't use this.  Returns the number of bytes sent, or 0 if the channel
        stream is closed.  Applications are responsible for checking that all
        data has been sent: if only some of the data was transmitted, the
        application needs to attempt delivery of the remaining data.

        :param bytes s: data to send.
        :return: number of bytes actually sent, as an `int`.

        :raises socket.timeout:
            if no data could be sent before the timeout set by `settimeout`.

        .. versionadded:: 1.1
        """
    m = Message()
    m.add_byte(cMSG_CHANNEL_EXTENDED_DATA)
    m.add_int(self.remote_chanid)
    m.add_int(1)
    return self._send(s, m)