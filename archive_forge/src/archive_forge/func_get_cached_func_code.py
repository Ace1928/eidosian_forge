from pickle import PicklingError
import re
import os
import os.path
import datetime
import json
import shutil
import warnings
import collections
import operator
import threading
from abc import ABCMeta, abstractmethod
from .backports import concurrency_safe_rename
from .disk import mkdirp, memstr_to_bytes, rm_subdirs
from . import numpy_pickle
def get_cached_func_code(self, path):
    """Store the code of the cached function."""
    path += ['func_code.py']
    filename = os.path.join(self.location, *path)
    try:
        with self._open_item(filename, 'rb') as f:
            return f.read().decode('utf-8')
    except:
        raise