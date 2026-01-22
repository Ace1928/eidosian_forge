from contextlib import contextmanager
import errno
import functools
import logging
import os
from pathlib import Path
import threading
import time
from typing import Callable
from typing import Optional
from typing import Union
from fasteners import _utils
from fasteners.process_mechanism import _interprocess_mechanism
from fasteners.process_mechanism import _interprocess_reader_writer_mechanism
def interprocess_read_locked(path: Union[Path, str]):
    """Acquires & releases an interprocess **read** lock around the call into
    the decorated function

    Args:
        path: Path to the file used for locking.
    """
    lock = InterProcessReaderWriterLock(path)

    def decorator(f):

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            with lock.read_lock():
                return f(*args, **kwargs)
        return wrapper
    return decorator