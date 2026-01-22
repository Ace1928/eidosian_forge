import os
import sys
from paramiko.sftp import SFTP_OP_UNSUPPORTED

        Create a symbolic link on the server, as new pathname ``path``,
        with ``target_path`` as the target of the link.

        :param str target_path:
            path (relative or absolute) of the target for this new symbolic
            link.
        :param str path:
            path (relative or absolute) of the symbolic link to create.
        :return: an error code `int` like ``SFTP_OK``.
        