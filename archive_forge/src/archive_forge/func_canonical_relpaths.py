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
def canonical_relpaths(base, paths):
    """Create an iterable to canonicalize a sequence of relative paths.

    The intent is for this implementation to use a cache, vastly speeding
    up multiple transformations in the same directory.
    """
    return [canonical_relpath(base, p) for p in paths]