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
def test_manager_should_propagate_exceptions(self):
    em = ExtensionManager.make_test_instance([test_extension], propagate_map_exceptions=True)
    self.skipTest('Skipping temporarily')
    func = Mock(side_effect=RuntimeError('hard coded error'))
    em.map(func, 1, 2, a='A', b='B')