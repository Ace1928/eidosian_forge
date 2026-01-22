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
def load_item(self, path, verbose=1, msg=None):
    """Load an item from the store given its path as a list of
           strings."""
    full_path = os.path.join(self.location, *path)
    if verbose > 1:
        if verbose < 10:
            print('{0}...'.format(msg))
        else:
            print('{0} from {1}'.format(msg, full_path))
    mmap_mode = None if not hasattr(self, 'mmap_mode') else self.mmap_mode
    filename = os.path.join(full_path, 'output.pkl')
    if not self._item_exists(filename):
        raise KeyError('Non-existing item (may have been cleared).\nFile %s does not exist' % filename)
    if mmap_mode is None:
        with self._open_item(filename, 'rb') as f:
            item = numpy_pickle.load(f)
    else:
        item = numpy_pickle.load(filename, mmap_mode=mmap_mode)
    return item