import os.path
import sys
import pathlib
import unittest
from importlib import import_module
from importlib_resources.readers import MultiplexedPath, NamespaceReader
def test_join_path(self):
    data01 = pathlib.Path(__file__).parent.joinpath('data01')
    prefix = str(data01.parent)
    path = MultiplexedPath(self.folder, data01)
    self.assertEqual(str(path.joinpath('binary.file'))[len(prefix) + 1:], os.path.join('namespacedata01', 'binary.file'))
    sub = path.joinpath('subdirectory')
    assert isinstance(sub, MultiplexedPath)
    assert 'namespacedata01' in str(sub)
    assert 'data01' in str(sub)
    self.assertEqual(str(path.joinpath('imaginary'))[len(prefix) + 1:], os.path.join('namespacedata01', 'imaginary'))
    self.assertEqual(path.joinpath(), path)