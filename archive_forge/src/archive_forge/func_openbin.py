from __future__ import absolute_import, unicode_literals
import typing
import contextlib
import io
import os
import six
import time
from collections import OrderedDict
from threading import RLock
from . import errors
from ._typing import overload
from .base import FS
from .copy import copy_modified_time
from .enums import ResourceType, Seek
from .info import Info
from .mode import Mode
from .path import iteratepath, normpath, split
def openbin(self, path, mode='r', buffering=-1, **options):
    _mode = Mode(mode)
    _mode.validate_bin()
    _path = self.validatepath(path)
    dir_path, file_name = split(_path)
    if not file_name:
        raise errors.FileExpected(path)
    with self._lock:
        parent_dir_entry = self._get_dir_entry(dir_path)
        if parent_dir_entry is None or not parent_dir_entry.is_dir:
            raise errors.ResourceNotFound(path)
        if _mode.create:
            if file_name not in parent_dir_entry:
                file_dir_entry = self._make_dir_entry(ResourceType.file, file_name)
                parent_dir_entry.set_entry(file_name, file_dir_entry)
            else:
                file_dir_entry = self._get_dir_entry(_path)
                if _mode.exclusive:
                    raise errors.FileExists(path)
            if file_dir_entry.is_dir:
                raise errors.FileExpected(path)
            mem_file = _MemoryFile(path=_path, memory_fs=self, mode=mode, dir_entry=file_dir_entry)
            file_dir_entry.add_open_file(mem_file)
            return mem_file
        if file_name not in parent_dir_entry:
            raise errors.ResourceNotFound(path)
        file_dir_entry = parent_dir_entry.get_entry(file_name)
        if file_dir_entry.is_dir:
            raise errors.FileExpected(path)
        mem_file = _MemoryFile(path=_path, memory_fs=self, mode=mode, dir_entry=file_dir_entry)
        file_dir_entry.add_open_file(mem_file)
        return mem_file