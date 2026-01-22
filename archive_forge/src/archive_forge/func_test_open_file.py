import os.path
import sys
import pathlib
import unittest
from importlib import import_module
from importlib_resources.readers import MultiplexedPath, NamespaceReader
def test_open_file(self):
    path = MultiplexedPath(self.folder)
    with self.assertRaises(FileNotFoundError):
        path.read_bytes()
    with self.assertRaises(FileNotFoundError):
        path.read_text()
    with self.assertRaises(FileNotFoundError):
        path.open()