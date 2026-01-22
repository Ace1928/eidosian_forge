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
def is_cwd(path: Union[Path, str]) -> bool:
    """Check whether a path is the current working directory.

    path (Union[Path, str]): The directory path.
    RETURNS (bool): Whether the path is the current working directory.
    """
    return str(Path(path).resolve()).lower() == str(Path.cwd().resolve()).lower()