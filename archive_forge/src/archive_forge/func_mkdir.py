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
def mkdir(self, path, create_parents=True, **kwargs):
    path = self._strip_protocol(path)
    if self.exists(path):
        raise FileExistsError(path)
    if create_parents:
        self.makedirs(path, exist_ok=True)
    else:
        os.mkdir(path, **kwargs)