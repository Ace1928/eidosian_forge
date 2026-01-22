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
def test_dispatch_map_should_invoke_filter_for_extensions(self):
    em = DispatchExtensionManager.make_test_instance([test_extension, test_extension2])
    filter_func = Mock(return_value=False)
    args = ('A',)
    kw = {'big': 'Cheese'}
    em.map(filter_func, None, *args, **kw)
    filter_func.assert_any_call(test_extension, *args, **kw)
    filter_func.assert_any_call(test_extension2, *args, **kw)