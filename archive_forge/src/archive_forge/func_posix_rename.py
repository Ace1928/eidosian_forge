import os
import sys
from paramiko.sftp import SFTP_OP_UNSUPPORTED
def posix_rename(self, oldpath, newpath):
    """
        Rename (or move) a file, following posix conventions. If newpath
        already exists, it will be overwritten.

        :param str oldpath:
            the requested path (relative or absolute) of the existing file.
        :param str newpath: the requested new path of the file.
        :return: an SFTP error code `int` like ``SFTP_OK``.

        :versionadded: 2.2
        """
    return SFTP_OP_UNSUPPORTED