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
def is_inside_or_parent_of_any(dir_list, fname):
    """True if fname is a child or a parent of any of the given files."""
    for dirname in dir_list:
        if is_inside(dirname, fname) or is_inside(fname, dirname):
            return True
    return False