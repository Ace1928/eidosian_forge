import collections
import random
import threading
import time
from concurrent import futures
import fasteners
from fasteners import test
from fasteners import _utils
def test_writer_to_reader(self):
    lock = fasteners.ReaderWriterLock()

    def reader_func():
        with lock.read_lock():
            self.assertTrue(lock.is_writer())
            self.assertTrue(lock.is_reader())
    with lock.write_lock():
        self.assertIsNone(reader_func())
        self.assertFalse(lock.is_reader())
    self.assertFalse(lock.is_reader())
    self.assertFalse(lock.is_writer())