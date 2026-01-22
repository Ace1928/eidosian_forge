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
def test_manager_should_eat_exceptions(self):
    em = ExtensionManager.make_test_instance([test_extension])
    func = Mock(side_effect=RuntimeError('hard coded error'))
    results = em.map(func, 1, 2, a='A', b='B')
    self.assertEqual(results, [])