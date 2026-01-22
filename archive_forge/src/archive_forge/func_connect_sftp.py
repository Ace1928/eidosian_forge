import errno
import getpass
import logging
import os
import socket
import subprocess
import sys
from binascii import hexlify
from typing import Dict, Optional, Set, Tuple, Type
from .. import bedding, config, errors, osutils, trace, ui
import weakref
def connect_sftp(self, username, password, host, port):
    try:
        argv = self._get_vendor_specific_argv(username, host, port, subsystem='sftp')
        sock = self._connect(argv)
        return SFTPClient(SocketAsChannelAdapter(sock))
    except _ssh_connection_errors as e:
        self._raise_connection_error(host, port=port, orig_error=e)