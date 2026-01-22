import io
import unittest
import importlib_resources as resources
from importlib_resources._adapters import (
from . import util
def test_orphan_path_name(self):
    self.assertEqual((self.files / 'a' / 'b').name, 'b')
    self.assertEqual((self.files / 'a' / 'b' / 'c').name, 'c')