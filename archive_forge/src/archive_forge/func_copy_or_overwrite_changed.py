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
def copy_or_overwrite_changed(source_path: StrPath, target_path: StrPath) -> StrPath:
    """Copy source_path to target_path, unless it already exists with the same mtime.

    We liberally add write permissions to deal with the case of multiple users needing
    to share the same cache or run directory.

    Args:
        source_path: The path to the file to copy.
        target_path: The path to copy the file to.

    Returns:
        The path to the copied file (which may be different from target_path).
    """
    return_type = type(target_path)
    target_path = system_preferred_path(target_path, warn=True)
    need_copy = not os.path.isfile(target_path) or os.stat(source_path).st_mtime != os.stat(target_path).st_mtime
    permissions_plus_write = os.stat(source_path).st_mode
    if need_copy:
        dir_name, file_name = os.path.split(target_path)
        target_path = os.path.join(mkdir_allow_fallback(dir_name), file_name)
        try:
            shutil.copy2(source_path, target_path)
        except PermissionError:
            try:
                os.chmod(target_path, permissions_plus_write)
                shutil.copy2(source_path, target_path)
            except PermissionError as e:
                raise PermissionError("Unable to overwrite '{target_path!s}'") from e
        os.chmod(target_path, permissions_plus_write)
    return return_type(target_path)