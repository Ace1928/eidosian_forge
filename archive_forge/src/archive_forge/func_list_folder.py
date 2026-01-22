import os
import socket
import socketserver
import sys
import time
import paramiko
from .. import osutils, trace, urlutils
from ..transport import ssh
from . import test_server
def list_folder(self, path):
    path = self._realpath(path)
    try:
        out = []
        if sys.platform == 'win32':
            flist = [f.encode('utf8') for f in os.listdir(path)]
        else:
            flist = os.listdir(path)
        for fname in flist:
            attr = paramiko.SFTPAttributes.from_stat(os.stat(osutils.pathjoin(path, fname)))
            attr.filename = fname
            out.append(attr)
        return out
    except OSError as e:
        return paramiko.SFTPServer.convert_errno(e.errno)