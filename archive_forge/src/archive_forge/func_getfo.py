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
def getfo(self, remotepath, fl, callback=None, prefetch=True, max_concurrent_prefetch_requests=None):
    """
        Copy a remote file (``remotepath``) from the SFTP server and write to
        an open file or file-like object, ``fl``.  Any exception raised by
        operations will be passed through.  This method is primarily provided
        as a convenience.

        :param object remotepath: opened file or file-like object to copy to
        :param str fl:
            the destination path on the local host or open file object
        :param callable callback:
            optional callback function (form: ``func(int, int)``) that accepts
            the bytes transferred so far and the total bytes to be transferred
        :param bool prefetch:
            controls whether prefetching is performed (default: True)
        :param int max_concurrent_prefetch_requests:
            The maximum number of concurrent read requests to prefetch. See
            `.SFTPClient.get` (its ``max_concurrent_prefetch_requests`` param)
            for details.
        :return: the `number <int>` of bytes written to the opened file object

        .. versionadded:: 1.10
        .. versionchanged:: 2.8
            Added the ``prefetch`` keyword argument.
        .. versionchanged:: 3.3
            Added ``max_concurrent_prefetch_requests``.
        """
    file_size = self.stat(remotepath).st_size
    with self.open(remotepath, 'rb') as fr:
        if prefetch:
            fr.prefetch(file_size, max_concurrent_prefetch_requests)
        return self._transfer_with_callback(reader=fr, writer=fl, file_size=file_size, callback=callback)