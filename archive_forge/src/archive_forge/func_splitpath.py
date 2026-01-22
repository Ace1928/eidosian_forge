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
def splitpath(p):
    """Turn string into list of parts."""
    use_bytes = isinstance(p, bytes)
    if os.path.sep == '\\':
        if use_bytes:
            ps = re.split(b'[\\\\/]', p)
        else:
            ps = re.split('[\\\\/]', p)
    elif use_bytes:
        ps = p.split(b'/')
    else:
        ps = p.split('/')
    if use_bytes:
        parent_dir = b'..'
        current_empty_dir = (b'.', b'')
    else:
        parent_dir = '..'
        current_empty_dir = ('.', '')
    rps = []
    for f in ps:
        if f == parent_dir:
            raise errors.BzrError(gettext('sorry, %r not allowed in path') % f)
        elif f in current_empty_dir:
            pass
        else:
            rps.append(f)
    return rps