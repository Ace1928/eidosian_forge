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
def test_hook_manager_should_return_named_extensions(self):
    hook1 = Extension('captain', None, None, None)
    hook2 = Extension('captain', None, None, None)
    em = HookManager.make_test_instance([hook1, hook2])
    self.assertEqual([hook1, hook2], em['captain'])