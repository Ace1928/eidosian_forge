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
def normalizepath(f):
    if getattr(os.path, 'realpath', None) is not None:
        F = realpath
    else:
        F = abspath
    [p, e] = os.path.split(f)
    if e == '' or e == '.' or e == '..':
        return F(f)
    else:
        return pathjoin(F(p), e)