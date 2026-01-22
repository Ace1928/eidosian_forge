import os.path
import sys
import pathlib
import unittest
from importlib import import_module
from importlib_resources.readers import MultiplexedPath, NamespaceReader
def test_iterdir(self):
    contents = {path.name for path in MultiplexedPath(self.folder).iterdir()}
    try:
        contents.remove('__pycache__')
    except (KeyError, ValueError):
        pass
    self.assertEqual(contents, {'subdirectory', 'binary.file', 'utf-16.file', 'utf-8.file'})