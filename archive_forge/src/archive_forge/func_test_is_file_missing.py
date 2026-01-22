import sys
import unittest
import importlib_resources as resources
import pathlib
from . import data01
from . import util
from importlib import import_module
def test_is_file_missing(self):
    target = resources.files(self.data) / 'not-a-file'
    self.assertFalse(target.is_file())