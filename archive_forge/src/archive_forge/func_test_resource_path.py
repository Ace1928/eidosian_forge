import os.path
import sys
import pathlib
import unittest
from importlib import import_module
from importlib_resources.readers import MultiplexedPath, NamespaceReader
def test_resource_path(self):
    namespacedata01 = import_module('namespacedata01')
    reader = NamespaceReader(namespacedata01.__spec__.submodule_search_locations)
    root = os.path.abspath(os.path.join(__file__, '..', 'namespacedata01'))
    self.assertEqual(reader.resource_path('binary.file'), os.path.join(root, 'binary.file'))
    self.assertEqual(reader.resource_path('imaginary'), os.path.join(root, 'imaginary'))