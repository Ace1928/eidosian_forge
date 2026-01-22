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
@open_only
def get_pty(self, term='vt100', width=80, height=24, width_pixels=0, height_pixels=0):
    """
        Request a pseudo-terminal from the server.  This is usually used right
        after creating a client channel, to ask the server to provide some
        basic terminal semantics for a shell invoked with `invoke_shell`.
        It isn't necessary (or desirable) to call this method if you're going
        to execute a single command with `exec_command`.

        :param str term: the terminal type to emulate
            (for example, ``'vt100'``)
        :param int width: width (in characters) of the terminal screen
        :param int height: height (in characters) of the terminal screen
        :param int width_pixels: width (in pixels) of the terminal screen
        :param int height_pixels: height (in pixels) of the terminal screen

        :raises:
            `.SSHException` -- if the request was rejected or the channel was
            closed
        """
    m = Message()
    m.add_byte(cMSG_CHANNEL_REQUEST)
    m.add_int(self.remote_chanid)
    m.add_string('pty-req')
    m.add_boolean(True)
    m.add_string(term)
    m.add_int(width)
    m.add_int(height)
    m.add_int(width_pixels)
    m.add_int(height_pixels)
    m.add_string(bytes())
    self._event_pending()
    self.transport._send_user_message(m)
    self._wait_for_event()