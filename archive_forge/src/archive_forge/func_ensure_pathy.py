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
def ensure_pathy(path):
    """Temporary helper to prevent importing cloudpathlib globally (which was
    originally added due to a slow and annoying Google Cloud warning with
    Pathy)"""
    from cloudpathlib import AnyPath
    return AnyPath(path)