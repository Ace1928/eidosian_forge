import collections
import random
import threading
import time
from concurrent import futures
import fasteners
from fasteners import test
from fasteners import _utils
def test_reader_to_writer(self):
    lock = fasteners.ReaderWriterLock()

    def writer_func():
        with lock.write_lock():
            pass
    with lock.read_lock():
        self.assertRaises(RuntimeError, writer_func)
        self.assertFalse(lock.is_writer())
    self.assertFalse(lock.is_reader())
    self.assertFalse(lock.is_writer())