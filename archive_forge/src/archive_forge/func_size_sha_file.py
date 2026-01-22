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
def size_sha_file(f):
    """Calculate the size and hexdigest of an open file.

    The file cursor should be already at the start and
    the caller is responsible for closing the file afterwards.
    """
    size = 0
    s = sha()
    BUFSIZE = 128 << 10
    while True:
        b = f.read(BUFSIZE)
        if not b:
            break
        size += len(b)
        s.update(b)
    return (size, _hexdigest(s))