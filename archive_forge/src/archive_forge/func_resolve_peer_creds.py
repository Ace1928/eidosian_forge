import os
import io
import re
import email.utils
import socket
import sys
import time
import traceback as traceback_
import logging
import platform
import queue
import contextlib
import threading
import urllib.parse
from functools import lru_cache
from . import connections, errors, __version__
from ._compat import bton
from ._compat import IS_PPC
from .workers import threadpool
from .makefile import MakeFile, StreamWriter
def resolve_peer_creds(self):
    """Look up the username and group tuple of the ``PEERCREDS``.

        :returns: the username and group tuple of the ``PEERCREDS``

        :raises NotImplementedError: if the OS is unsupported
        :raises RuntimeError: if UID/GID lookup is unsupported or disabled
        """
    if not IS_UID_GID_RESOLVABLE:
        raise NotImplementedError('UID/GID lookup is unavailable under current platform. It can only be done under UNIX-like OS but not under the Google App Engine')
    elif not self.peercreds_resolve_enabled:
        raise RuntimeError('UID/GID lookup is disabled within this server')
    user = pwd.getpwuid(self.peer_uid).pw_name
    group = grp.getgrgid(self.peer_gid).gr_name
    return (user, group)