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
def compare_paths_prefix_order(path_a, path_b):
    """Compare path_a and path_b to generate the same order walkdirs uses."""
    key_a = path_prefix_key(path_a)
    key_b = path_prefix_key(path_b)
    return (key_a > key_b) - (key_a < key_b)