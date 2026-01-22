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
def is_exec(fpath: str) -> bool:
    return osp.isfile(fpath) and os.access(fpath, os.X_OK) and (os.name != 'nt' or not winprog_exts or any((fpath.upper().endswith(ext) for ext in winprog_exts)))