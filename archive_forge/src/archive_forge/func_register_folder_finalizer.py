from mmap import mmap
import errno
import os
import stat
import threading
import atexit
import tempfile
import time
import warnings
import weakref
from uuid import uuid4
from multiprocessing import util
from pickle import whichmodule, loads, dumps, HIGHEST_PROTOCOL, PicklingError
from .numpy_pickle import dump, load, load_temporary_memmap
from .backports import make_memmap
from .disk import delete_folder
from .externals.loky.backend import resource_tracker
def register_folder_finalizer(self, pool_subfolder, context_id):
    pool_module_name = whichmodule(delete_folder, 'delete_folder')
    resource_tracker.register(pool_subfolder, 'folder')

    def _cleanup():
        delete_folder = __import__(pool_module_name, fromlist=['delete_folder']).delete_folder
        try:
            delete_folder(pool_subfolder, allow_non_empty=True)
            resource_tracker.unregister(pool_subfolder, 'folder')
        except OSError:
            warnings.warn('Failed to delete temporary folder: {}'.format(pool_subfolder))
    self._finalizers[context_id] = atexit.register(_cleanup)