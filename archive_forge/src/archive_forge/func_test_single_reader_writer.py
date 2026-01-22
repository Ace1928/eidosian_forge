import collections
import random
import threading
import time
from concurrent import futures
import fasteners
from fasteners import test
from fasteners import _utils
def test_single_reader_writer(self):
    results = []
    lock = fasteners.ReaderWriterLock()
    with lock.read_lock():
        self.assertTrue(lock.is_reader())
        self.assertEqual(0, len(results))
    with lock.write_lock():
        results.append(1)
        self.assertTrue(lock.is_writer())
    with lock.read_lock():
        self.assertTrue(lock.is_reader())
        self.assertEqual(1, len(results))
    self.assertFalse(lock.is_reader())
    self.assertFalse(lock.is_writer())