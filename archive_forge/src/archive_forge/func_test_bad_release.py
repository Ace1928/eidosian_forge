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
def test_bad_release(self):
    lock_file = os.path.join(self.lock_dir, 'lock')
    lock = pl.InterProcessLock(lock_file)
    self.assertRaises(threading.ThreadError, lock.release)