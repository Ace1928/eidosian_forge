import os
import socket
import socketserver
import sys
import time
import paramiko
from .. import osutils, trace, urlutils
from ..transport import ssh
from . import test_server
def chattr(self, path, attr):
    try:
        paramiko.SFTPServer.set_file_attr(path, attr)
    except OSError as e:
        return paramiko.SFTPServer.convert_errno(e.errno)
    return paramiko.SFTP_OK