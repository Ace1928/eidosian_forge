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
def assure_directory_exists(path: PathLike, is_file: bool=False) -> bool:
    """Make sure that the directory pointed to by path exists.

    :param is_file: If True, ``path`` is assumed to be a file and handled correctly.
        Otherwise it must be a directory.

    :return: True if the directory was created, False if it already existed.
    """
    if is_file:
        path = osp.dirname(path)
    if not osp.isdir(path):
        os.makedirs(path, exist_ok=True)
        return True
    return False