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
def put_file(self, path1, path2, callback=None, **kwargs):
    return self.cp_file(path1, path2, **kwargs)