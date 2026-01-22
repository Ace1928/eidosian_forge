import os
import shutil
import stat
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterator, Union
from ..errors import Warnings
def is_subpath_of(parent, child):
    """
    Check whether `child` is a path contained within `parent`.
    """
    parent_realpath = os.path.realpath(parent)
    child_realpath = os.path.realpath(child)
    return os.path.commonpath([parent_realpath, child_realpath]) == parent_realpath