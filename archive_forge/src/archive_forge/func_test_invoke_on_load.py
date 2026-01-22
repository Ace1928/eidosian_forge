import importlib.metadata as importlib_metadata
import operator
from unittest import mock
from stevedore import exception
from stevedore import extension
from stevedore.tests import utils
def test_invoke_on_load(self):
    em = extension.ExtensionManager('stevedore.test.extension', invoke_on_load=True, invoke_args=('a',), invoke_kwds={'b': 'B'})
    self.assertEqual(len(em.extensions), 2)
    for e in em.extensions:
        self.assertEqual(e.obj.args, ('a',))
        self.assertEqual(e.obj.kwds, {'b': 'B'})