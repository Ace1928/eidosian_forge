from __future__ import annotations
import abc
import collections.abc as c
import enum
import fcntl
import importlib.util
import inspect
import json
import keyword
import os
import platform
import pkgutil
import random
import re
import shutil
import stat
import string
import subprocess
import sys
import time
import functools
import shlex
import typing as t
import warnings
from struct import unpack, pack
from termios import TIOCGWINSZ
from .locale_util import (
from .encoding import (
from .io import (
from .thread import (
from .constants import (
def is_binary_file(path: str) -> bool:
    """Return True if the specified file is a binary file, otherwise return False."""
    assume_text = {'.cfg', '.conf', '.crt', '.cs', '.css', '.html', '.ini', '.j2', '.js', '.json', '.md', '.pem', '.ps1', '.psm1', '.py', '.rst', '.sh', '.txt', '.xml', '.yaml', '.yml'}
    assume_binary = {'.bin', '.eot', '.gz', '.ico', '.iso', '.jpg', '.otf', '.p12', '.png', '.pyc', '.rpm', '.ttf', '.woff', '.woff2', '.zip'}
    ext = os.path.splitext(path)[1]
    if ext in assume_text:
        return False
    if ext in assume_binary:
        return True
    with open_binary_file(path) as path_fd:
        return b'\x00' in path_fd.read(4096)