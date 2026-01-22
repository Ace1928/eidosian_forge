import io
import unittest
import importlib_resources as resources
from importlib_resources._adapters import (
from . import util
def test_orphan_path_open(self):
    with self.assertRaises(FileNotFoundError):
        (self.files / 'a' / 'b').read_bytes()
    with self.assertRaises(FileNotFoundError):
        (self.files / 'a' / 'b' / 'c').read_bytes()