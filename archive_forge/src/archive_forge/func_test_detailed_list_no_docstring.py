import importlib.metadata as importlib_metadata
from stevedore import extension
from stevedore import sphinxext
from stevedore.tests import utils
def test_detailed_list_no_docstring(self):
    ext = [_make_ext('nodoc', None)]
    em = extension.ExtensionManager.make_test_instance(ext)
    results = list(sphinxext._detailed_list(em))
    self.assertEqual([('nodoc', 'nodoc_module'), ('-----', 'nodoc_module'), ('\n', 'nodoc_module'), ('.. warning:: No documentation found for nodoc in nodoc_module:nodoc', 'nodoc_module'), ('\n', 'nodoc_module')], results)