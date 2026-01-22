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
def join_path(a: PathLike, *p: PathLike) -> PathLike:
    """Join path tokens together similar to osp.join, but always use
    '/' instead of possibly '\\' on Windows."""
    path = str(a)
    for b in p:
        b = str(b)
        if not b:
            continue
        if b.startswith('/'):
            path += b[1:]
        elif path == '' or path.endswith('/'):
            path += b
        else:
            path += '/' + b
    return path