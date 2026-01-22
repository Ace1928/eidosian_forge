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
def dereference_path(path):
    """Determine the real path to a file.

    All parent elements are dereferenced.  But the file itself is not
    dereferenced.
    :param path: The original path.  May be absolute or relative.
    :return: the real path *to* the file
    """
    parent, base = os.path.split(path)
    return pathjoin(realpath(pathjoin('.', parent)), base)