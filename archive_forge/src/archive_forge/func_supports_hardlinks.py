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
def supports_hardlinks(path):
    if getattr(os, 'link', None) is None:
        return False
    try:
        fs_type = get_fs_type(path)
    except errors.DependencyNotPresent as e:
        trace.mutter('Unable to get fs type for %r: %s', path, e)
        return True
    else:
        if fs_type is None:
            return sys.platform != 'win32'
        if fs_type in ('vfat', 'ntfs'):
            return False
        return True