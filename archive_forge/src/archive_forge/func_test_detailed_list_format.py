import importlib.metadata as importlib_metadata
from stevedore import extension
from stevedore import sphinxext
from stevedore.tests import utils
def test_detailed_list_format(self):
    results = list(sphinxext._detailed_list(self.em, over='+', under='+'))
    self.assertEqual([('+++++', 'test1_module'), ('test1', 'test1_module'), ('+++++', 'test1_module'), ('\n', 'test1_module'), ('One-line docstring', 'test1_module'), ('\n', 'test1_module'), ('+++++', 'test2_module'), ('test2', 'test2_module'), ('+++++', 'test2_module'), ('\n', 'test2_module'), ('Multi-line docstring\n\nAnother para', 'test2_module'), ('\n', 'test2_module')], results)