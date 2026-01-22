import unittest
import importlib_resources as resources
from . import data01
from . import util
from importlib import import_module
def test_read_submodule_resource(self):
    submodule = import_module('namespacedata01.subdirectory')
    result = resources.files(submodule).joinpath('binary.file').read_bytes()
    self.assertEqual(result, bytes(range(12, 16)))