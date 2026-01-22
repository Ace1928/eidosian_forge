import importlib.metadata as importlib_metadata
from stevedore import extension
from stevedore import sphinxext
from stevedore.tests import utils
def test_simple_list(self):
    results = list(sphinxext._simple_list(self.em))
    self.assertEqual([('* test1 -- One-line docstring', 'test1_module'), ('* test2 -- Multi-line docstring', 'test2_module')], results)