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
def unlink_file(filename):
    """Wrapper around os.unlink with a retry mechanism.

    The retry mechanism has been implemented primarily to overcome a race
    condition happening during the finalizer of a np.memmap: when a process
    holding the last reference to a mmap-backed np.memmap/np.array is about to
    delete this array (and close the reference), it sends a maybe_unlink
    request to the resource_tracker. This request can be processed faster than
    it takes for the last reference of the memmap to be closed, yielding (on
    Windows) a PermissionError in the resource_tracker loop.
    """
    NUM_RETRIES = 10
    for retry_no in range(1, NUM_RETRIES + 1):
        try:
            os.unlink(filename)
            break
        except PermissionError:
            util.debug('[ResourceTracker] tried to unlink {}, got PermissionError'.format(filename))
            if retry_no == NUM_RETRIES:
                raise
            else:
                time.sleep(0.2)
        except FileNotFoundError:
            pass