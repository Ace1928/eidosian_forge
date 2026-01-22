import os
import shutil
import tempfile
import threading
import unittest
from traits.util.event_tracer import (
def test_record_container(self):
    container = RecordContainer()
    for i in range(7):
        container.record(SentinelRecord())
    self.assertEqual(len(container._records), 7)
    container.save_to_file(self.filename)
    with open(self.filename, 'r', encoding='utf-8') as handle:
        lines = handle.readlines()
    self.assertEqual(lines, ['\n'] * 7)