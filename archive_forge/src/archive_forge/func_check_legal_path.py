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
def check_legal_path(path):
    """Check whether the supplied path is legal.
    This is only required on Windows, so we don't test on other platforms
    right now.
    """
    if sys.platform != 'win32':
        return
    if _validWin32PathRE.match(path) is None:
        raise errors.IllegalPath(path)