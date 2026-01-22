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
def minimum_path_selection(paths):
    """Return the smallset subset of paths which are outside paths.

    :param paths: A container (and hence not None) of paths.
    :return: A set of paths sufficient to include everything in paths via
        is_inside, drawn from the paths parameter.
    """
    if len(paths) < 2:
        return set(paths)

    def sort_key(path):
        if isinstance(path, bytes):
            return path.split(b'/')
        else:
            return path.split('/')
    sorted_paths = sorted(list(paths), key=sort_key)
    search_paths = [sorted_paths[0]]
    for path in sorted_paths[1:]:
        if not is_inside(search_paths[-1], path):
            search_paths.append(path)
    return set(search_paths)