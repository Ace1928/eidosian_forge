from binascii import hexlify
import errno
import os
import stat
import threading
import time
import weakref
from paramiko import util
from paramiko.channel import Channel
from paramiko.message import Message
from paramiko.common import INFO, DEBUG, o777
from paramiko.sftp import (
from paramiko.sftp_attr import SFTPAttributes
from paramiko.ssh_exception import SSHException
from paramiko.sftp_file import SFTPFile
from paramiko.util import ClosingContextManager, b, u
def listdir_attr(self, path='.'):
    """
        Return a list containing `.SFTPAttributes` objects corresponding to
        files in the given ``path``.  The list is in arbitrary order.  It does
        not include the special entries ``'.'`` and ``'..'`` even if they are
        present in the folder.

        The returned `.SFTPAttributes` objects will each have an additional
        field: ``longname``, which may contain a formatted string of the file's
        attributes, in unix format.  The content of this string will probably
        depend on the SFTP server implementation.

        :param str path: path to list (defaults to ``'.'``)
        :return: list of `.SFTPAttributes` objects

        .. versionadded:: 1.2
        """
    path = self._adjust_cwd(path)
    self._log(DEBUG, 'listdir({!r})'.format(path))
    t, msg = self._request(CMD_OPENDIR, path)
    if t != CMD_HANDLE:
        raise SFTPError('Expected handle')
    handle = msg.get_binary()
    filelist = []
    while True:
        try:
            t, msg = self._request(CMD_READDIR, handle)
        except EOFError:
            break
        if t != CMD_NAME:
            raise SFTPError('Expected name response')
        count = msg.get_int()
        for i in range(count):
            filename = msg.get_text()
            longname = msg.get_text()
            attr = SFTPAttributes._from_msg(msg, filename, longname)
            if filename != '.' and filename != '..':
                filelist.append(attr)
    self._request(CMD_CLOSE, handle)
    return filelist