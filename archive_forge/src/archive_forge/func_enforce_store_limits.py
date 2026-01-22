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
def enforce_store_limits(self, bytes_limit, items_limit=None, age_limit=None):
    """
        Remove the store's oldest files to enforce item, byte, and age limits.
        """
    items_to_delete = self._get_items_to_delete(bytes_limit, items_limit, age_limit)
    for item in items_to_delete:
        if self.verbose > 10:
            print('Deleting item {0}'.format(item))
        try:
            self.clear_location(item.path)
        except OSError:
            pass