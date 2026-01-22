from abc import abstractmethod
import contextlib
from functools import wraps
import getpass
import logging
import os
import os.path as osp
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit
import warnings
from typing import (
from .types import (
from gitdb.util import (  # noqa: F401  # @IgnorePep8
def decygpath(path: PathLike) -> str:
    path = str(path)
    m = _decygpath_regex.match(path)
    if m:
        drive, rest_path = m.groups()
        path = '%s:%s' % (drive.upper(), rest_path or '')
    return path.replace('/', '\\')