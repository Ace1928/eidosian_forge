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
def clear_path(self, path):
    """Clear all items with a common path in the store."""
    func_path = os.path.join(self.location, *path)
    if self._item_exists(func_path):
        self.clear_location(func_path)