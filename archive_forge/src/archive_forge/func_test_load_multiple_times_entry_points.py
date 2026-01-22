import importlib.metadata as importlib_metadata
import operator
from unittest import mock
from stevedore import exception
from stevedore import extension
from stevedore.tests import utils
def test_load_multiple_times_entry_points(self):
    em1 = extension.ExtensionManager('stevedore.test.extension')
    eps1 = [ext.entry_point for ext in em1]
    em2 = extension.ExtensionManager('stevedore.test.extension')
    eps2 = [ext.entry_point for ext in em2]
    self.assertIs(eps1[0], eps2[0])