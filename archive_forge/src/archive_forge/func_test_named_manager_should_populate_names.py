from unittest.mock import Mock
from unittest.mock import sentinel
from stevedore.dispatch import DispatchExtensionManager
from stevedore.dispatch import NameDispatchExtensionManager
from stevedore.extension import Extension
from stevedore.tests import utils
from stevedore import DriverManager
from stevedore import EnabledExtensionManager
from stevedore import ExtensionManager
from stevedore import HookManager
from stevedore import NamedExtensionManager
def test_named_manager_should_populate_names(self):
    extensions = [test_extension, test_extension2]
    em = NamedExtensionManager.make_test_instance(extensions)
    self.assertEqual(em.names(), ['test_extension', 'another_one'])