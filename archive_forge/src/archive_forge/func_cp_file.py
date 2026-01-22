import datetime
import io
import logging
import os
import os.path as osp
import re
import shutil
import stat
import tempfile
from fsspec import AbstractFileSystem
from fsspec.compression import compr
from fsspec.core import get_compression
from fsspec.utils import isfilelike, stringify_path
def cp_file(self, path1, path2, **kwargs):
    path1 = self._strip_protocol(path1).rstrip('/')
    path2 = self._strip_protocol(path2).rstrip('/')
    if self.auto_mkdir:
        self.makedirs(self._parent(path2), exist_ok=True)
    if self.isfile(path1):
        shutil.copyfile(path1, path2)
    elif self.isdir(path1):
        self.mkdirs(path2, exist_ok=True)
    else:
        raise FileNotFoundError(path1)