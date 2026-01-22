from __future__ import annotations
import logging
import warnings
from pathlib import Path
from socket import socket
from typing import TYPE_CHECKING, Callable
from watchgod import DefaultWatcher
from uvicorn.config import Config
from uvicorn.supervisors.basereload import BaseReload
def should_watch_file(self, entry: DirEntry) -> bool:
    cached_result = self.watched_files.get(entry.path)
    if cached_result is not None:
        return cached_result
    entry_path = Path(entry)
    if entry_path.parent == Path.cwd() and Path.cwd() not in self.dirs_includes:
        self.watched_files[entry.path] = False
        return False
    for include_pattern in self.includes:
        if str(entry_path).endswith(include_pattern):
            self.watched_files[entry.path] = True
            return True
        if entry_path.match(include_pattern):
            for exclude_pattern in self.excludes:
                if entry_path.match(exclude_pattern):
                    self.watched_files[entry.path] = False
                    return False
            self.watched_files[entry.path] = True
            return True
    self.watched_files[entry.path] = False
    return False