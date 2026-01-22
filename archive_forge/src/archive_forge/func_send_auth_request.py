import weakref
import threading
import time
import re
from paramiko.common import (
from paramiko.message import Message
from paramiko.util import b, u
from paramiko.ssh_exception import (
from paramiko.server import InteractiveQuery
from paramiko.ssh_gss import GSSAuth, GSS_EXCEPTIONS
def send_auth_request(self, username, method, finish_message=None):
    """
        Submit a userauth request message & wait for response.

        Performs the transport message send call, sets self.auth_event, and
        will lock-n-block as necessary to both send, and wait for response to,
        the USERAUTH_REQUEST.

        Most callers will want to supply a callback to ``finish_message``,
        which accepts a Message ``m`` and may call mutator methods on it to add
        more fields.
        """
    self.auth_method = method
    self.username = username
    m = Message()
    m.add_byte(cMSG_USERAUTH_REQUEST)
    m.add_string(username)
    m.add_string('ssh-connection')
    m.add_string(method)
    finish_message(m)
    with self.transport.lock:
        self.transport._send_message(m)
    self.auth_event = threading.Event()
    return self.wait_for_response(self.auth_event)