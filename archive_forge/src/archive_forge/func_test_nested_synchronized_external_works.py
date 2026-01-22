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
def test_nested_synchronized_external_works(self):
    sentinel = object()

    @pl.interprocess_locked(os.path.join(self.lock_dir, 'test-lock-1'))
    def outer_lock():

        @pl.interprocess_locked(os.path.join(self.lock_dir, 'test-lock-2'))
        def inner_lock():
            return sentinel
        return inner_lock()
    self.assertEqual(sentinel, outer_lock())