import errno
import fcntl
import multiprocessing
import os
import shutil
import signal
import tempfile
import threading
import time
from fasteners import process_lock as pl
from fasteners import test
def test_lock_file_exists(self):
    lock_file = os.path.join(self.lock_dir, 'lock')

    @pl.interprocess_locked(lock_file)
    def foo():
        self.assertTrue(os.path.exists(lock_file))
    foo()