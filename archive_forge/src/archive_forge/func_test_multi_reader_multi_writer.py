import collections
import random
import threading
import time
from concurrent import futures
import fasteners
from fasteners import test
from fasteners import _utils
def test_multi_reader_multi_writer(self):
    writer_times, reader_times = _spawn_variation(10, 10)
    self.assertEqual(10, len(writer_times))
    self.assertEqual(10, len(reader_times))
    for start, stop in writer_times:
        self.assertEqual(0, _find_overlaps(reader_times, start, stop))
        self.assertEqual(1, _find_overlaps(writer_times, start, stop))
    for start, stop in reader_times:
        self.assertEqual(0, _find_overlaps(writer_times, start, stop))