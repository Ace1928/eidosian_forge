import collections
import random
import threading
import time
from concurrent import futures
import fasteners
from fasteners import test
from fasteners import _utils
def test_multi_reader_single_writer(self):
    writer_times, reader_times = _spawn_variation(9, 1)
    self.assertEqual(1, len(writer_times))
    self.assertEqual(9, len(reader_times))
    start, stop = writer_times[0]
    self.assertEqual(0, _find_overlaps(reader_times, start, stop))