import io
import unittest
import importlib_resources as resources
from importlib_resources._adapters import (
from . import util
def test_orphan_path_is(self):
    self.assertFalse((self.files / 'a' / 'a').is_file())
    self.assertFalse((self.files / 'a' / 'a').is_dir())
    self.assertFalse((self.files / 'a' / 'a' / 'a').is_file())
    self.assertFalse((self.files / 'a' / 'a' / 'a').is_dir())