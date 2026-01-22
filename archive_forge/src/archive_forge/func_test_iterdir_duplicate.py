import os.path
import sys
import pathlib
import unittest
from importlib import import_module
from importlib_resources.readers import MultiplexedPath, NamespaceReader
def test_iterdir_duplicate(self):
    data01 = pathlib.Path(__file__).parent.joinpath('data01')
    contents = {path.name for path in MultiplexedPath(self.folder, data01).iterdir()}
    for remove in ('__pycache__', '__init__.pyc'):
        try:
            contents.remove(remove)
        except (KeyError, ValueError):
            pass
    self.assertEqual(contents, {'__init__.py', 'binary.file', 'subdirectory', 'utf-16.file', 'utf-8.file'})