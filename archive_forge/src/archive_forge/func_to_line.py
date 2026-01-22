from base64 import encodebytes, decodebytes
import binascii
import os
import re
from collections.abc import MutableMapping
from hashlib import sha1
from hmac import HMAC
from paramiko.pkey import PKey, UnknownKeyType
from paramiko.util import get_logger, constant_time_bytes_eq, b, u
from paramiko.ssh_exception import SSHException
def to_line(self):
    """
        Returns a string in OpenSSH known_hosts file format, or None if
        the object is not in a valid state.  A trailing newline is
        included.
        """
    if self.valid:
        return '{} {} {}\n'.format(','.join(self.hostnames), self.key.get_name(), self.key.get_base64())
    return None