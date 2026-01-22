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
@contextlib.contextmanager
def safe_open(path: StrPath, mode: str='r', *args: Any, **kwargs: Any) -> Generator[IO, None, None]:
    """Open a file, ensuring any changes only apply atomically after close.

    This context manager ensures that even unsuccessful writes will not leave a "dirty"
    file or overwrite good data, and that all temp data is cleaned up.

    The semantics and behavior are intended to be nearly identical to the built-in
    open() function. Differences:
        - It creates any parent directories that don't exist, rather than raising.
        - In 'x' mode, it checks at the beginning AND end of the write and fails if the
            file exists either time.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if 'x' in mode and path.exists():
        raise FileExistsError(f'{path!s} already exists')
    if 'r' in mode and '+' not in mode:
        with path.open(mode, *args, **kwargs) as f:
            yield f
        return
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp_dir:
        tmp_path = Path(tmp_dir) / path.name
        if ('r' in mode or 'a' in mode) and path.exists():
            shutil.copy2(path, tmp_path)
        with tmp_path.open(mode, *args, **kwargs) as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        if 'x' in mode:
            os.link(tmp_path, path)
            os.unlink(tmp_path)
        else:
            tmp_path.replace(path)