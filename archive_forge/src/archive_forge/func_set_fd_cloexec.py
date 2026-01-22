import codecs
import errno
import os
import re
import stat
import sys
import time
from functools import partial
from typing import Dict, List
from .lazy_import import lazy_import
import locale
import ntpath
import posixpath
import select
import shutil
from shutil import rmtree
import socket
import subprocess
import unicodedata
from breezy import (
from breezy.i18n import gettext
from hashlib import md5
from hashlib import sha1 as sha
import breezy
from . import errors
def set_fd_cloexec(fd):
    """Set a Unix file descriptor's FD_CLOEXEC flag.  Do nothing if platform
    support for this is not available.
    """
    try:
        import fcntl
        old = fcntl.fcntl(fd, fcntl.F_GETFD)
        fcntl.fcntl(fd, fcntl.F_SETFD, old | fcntl.FD_CLOEXEC)
    except (ImportError, AttributeError):
        pass