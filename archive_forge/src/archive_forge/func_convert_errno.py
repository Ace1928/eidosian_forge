import os
import errno
import sys
from hashlib import md5, sha1
from paramiko import util
from paramiko.sftp import (
from paramiko.sftp_si import SFTPServerInterface
from paramiko.sftp_attr import SFTPAttributes
from paramiko.common import DEBUG
from paramiko.server import SubsystemHandler
from paramiko.util import b
from paramiko.sftp import (
from paramiko.sftp_handle import SFTPHandle
@staticmethod
def convert_errno(e):
    """
        Convert an errno value (as from an ``OSError`` or ``IOError``) into a
        standard SFTP result code.  This is a convenience function for trapping
        exceptions in server code and returning an appropriate result.

        :param int e: an errno code, as from ``OSError.errno``.
        :return: an `int` SFTP error code like ``SFTP_NO_SUCH_FILE``.
        """
    if e == errno.EACCES:
        return SFTP_PERMISSION_DENIED
    elif e == errno.ENOENT or e == errno.ENOTDIR:
        return SFTP_NO_SUCH_FILE
    else:
        return SFTP_FAILURE