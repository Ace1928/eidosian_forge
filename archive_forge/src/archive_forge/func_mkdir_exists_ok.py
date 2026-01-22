import contextlib
import ctypes
import errno
import logging
import os
import platform
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import IO, Any, BinaryIO, Generator, Optional
from wandb.sdk.lib.paths import StrPath
def mkdir_exists_ok(dir_name: StrPath) -> None:
    """Create `dir_name` and any parent directories if they don't exist.

    Raises:
        FileExistsError: if `dir_name` exists and is not a directory.
        PermissionError: if `dir_name` is not writable.
    """
    try:
        os.makedirs(dir_name, exist_ok=True)
    except FileExistsError as e:
        raise FileExistsError(f'{dir_name!s} exists and is not a directory') from e
    except PermissionError as e:
        raise PermissionError(f'{dir_name!s} is not writable') from e