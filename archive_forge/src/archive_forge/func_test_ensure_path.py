import contextlib
from taskflow import exceptions as exc
from taskflow.persistence import backends
from taskflow.persistence.backends import impl_memory
from taskflow import test
from taskflow.tests.unit.persistence import base
def test_ensure_path(self):
    fs = impl_memory.FakeFilesystem()
    pieces = ['a', 'b', 'c']
    path = '/' + '/'.join(pieces)
    fs.ensure_path(path)
    path = fs.root_path
    for i, p in enumerate(pieces):
        if i == 0:
            path += p
        else:
            path += '/' + p
        self.assertIsNone(fs[path])