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
def force_remove(rmfunc, path, ex):
    os.chmod(path, stat.S_IWRITE)
    rmfunc(path)