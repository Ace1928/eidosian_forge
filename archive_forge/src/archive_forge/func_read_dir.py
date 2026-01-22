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
def read_dir(self, prefix, top):
    """Read a single directory from a non-utf8 file system.

        top, and the abspath element in the output are unicode, all other paths
        are utf8. Local disk IO is done via unicode calls to listdir etc.

        This is currently the fallback code path when the filesystem encoding is
        not UTF-8. It may be better to implement an alternative so that we can
        safely handle paths that are not properly decodable in the current
        encoding.

        See DirReader.read_dir for details.
        """
    _utf8_encode = self._utf8_encode
    if prefix:
        relprefix = prefix + b'/'
    else:
        relprefix = b''
    top_slash = top + '/'
    dirblock = []
    append = dirblock.append
    for entry in os.scandir(safe_utf8(top)):
        name = os.fsdecode(entry.name)
        abspath = top_slash + name
        name_utf8 = _utf8_encode(name, 'surrogateescape')[0]
        statvalue = entry.stat(follow_symlinks=False)
        kind = file_kind_from_stat_mode(statvalue.st_mode)
        append((relprefix + name_utf8, name_utf8, kind, statvalue, abspath))
    return sorted(dirblock)