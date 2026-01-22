import os
import socket
import socketserver
import sys
import time
import paramiko
from .. import osutils, trace, urlutils
from ..transport import ssh
from . import test_server
def lstat(self, path):
    path = self._realpath(path)
    try:
        return paramiko.SFTPAttributes.from_stat(os.lstat(path))
    except OSError as e:
        return paramiko.SFTPServer.convert_errno(e.errno)