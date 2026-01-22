from __future__ import annotations
import logging
from pathlib import Path
from socket import socket
from typing import Callable, Iterator
from uvicorn.config import Config
from uvicorn.supervisors.basereload import BaseReload
def iter_py_files(self) -> Iterator[Path]:
    for reload_dir in self.config.reload_dirs:
        for path in list(reload_dir.rglob('*.py')):
            yield path.resolve()