import contextlib
from taskflow import exceptions as exc
from taskflow.persistence import backends
from taskflow.persistence.backends import impl_memory
from taskflow import test
from taskflow.tests.unit.persistence import base
def test_set_get_ls(self):
    fs = impl_memory.FakeFilesystem()
    fs['/d'] = 'd'
    fs['/c'] = 'c'
    fs['/d/b'] = 'db'
    self.assertEqual(2, len(fs.ls('/')))
    self.assertEqual(1, len(fs.ls('/d')))
    self.assertEqual('d', fs['/d'])
    self.assertEqual('c', fs['/c'])
    self.assertEqual('db', fs['/d/b'])